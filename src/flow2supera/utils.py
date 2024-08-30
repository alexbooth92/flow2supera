import os
import numpy as np
import time
import flow2supera
import ROOT
import yaml
from yaml import Loader
from larcv import larcv
from supera import supera
import cppyy

#from LarpixParser import event_parser as EventParser

def get_iomanager(outname):
    import tempfile
    cfg='''                                                                                                                                          
IOManager: {                                                                                                                                         
  Verbosity:   2                                                                                                                                     
  Name:        "IOManager"                                                                                                                           
  IOMode:      1                                                                                                                                     
  OutFileName: "%s"                                                                                                                                  
}                                                                                                                                                    
'''
    #f=open('tmp.cfg','w')                                                                                                                           
    f=tempfile.NamedTemporaryFile(mode='w')
    f.write(cfg % outname)
    f.flush()
    o = larcv.IOManager(f.name)
    o.initialize()
    f.close()
    return o


def larcv_meta(supera_meta):
    larcv_meta = larcv.Voxel3DMeta()

    larcv_meta.set(supera_meta.min_x(),supera_meta.min_y(),supera_meta.min_z(),
                   supera_meta.max_x(),supera_meta.max_y(),supera_meta.max_z(),
                   supera_meta.num_voxel_x(),supera_meta.num_voxel_y(),supera_meta.num_voxel_z())
    
    return larcv_meta

def larcv_particle(p):
    
    US2NS = 1.e3

    larp=larcv.Particle()
    
    larp.id              (p.part.id)
    larp.shape           (int(p.part.shape))
    
    # particle's info setter
    larp.track_id         (p.part.trackid)
    if hasattr(p.part, "genid") and hasattr(larp, "gen_id"):
        larp.gen_id(p.part.genid)
        
    larp.pdg_code         (p.part.pdg)
    larp.momentum         (p.part.px,p.part.py,p.part.pz)
    larp.end_momentum     (p.part.end_px,p.part.end_py,p.part.end_pz)
    larp.distance_travel  (p.part.dist_travel)
    
    vtx_dict = dict(position = p.part.vtx, 
                    end_position = p.part.end_pt, 
                    first_step = p.part.first_step, 
                    last_step = p.part.last_step,
                    parent_position = p.part.parent_vtx,
                    ancestor_position = p.part.ancestor_vtx,
                   )
    for key,item in vtx_dict.items():
        getattr(larp,key)(item.pos.x, item.pos.y, item.pos.z, item.time * US2NS)
    
    #larp.distance_travel ( double dist ) { _dist_travel = dist; }
    larp.energy_init      (p.part.energy_init)
    larp.energy_deposit   (p.energy.sum())
    larp.creation_process (p.part.process)
    larp.num_voxels       (p.energy.size())
    
    # parent info setter
    larp.parent_track_id (p.part.parent_trackid)
    larp.parent_pdg_code (p.part.parent_pdg)
    larp.parent_creation_process(p.part.parent_process)
    larp.parent_id       (p.part.parent_id)
    for cid in p.part.children_id:
        larp.children_id(cid)

    # ancestor info setter
    larp.ancestor_track_id (p.part.ancestor_id)
    larp.ancestor_pdg_code (p.part.ancestor_pdg)
    larp.ancestor_creation_process(p.part.ancestor_process)
                                   
    if not p.part.group_id == supera.kINVALID_INSTANCEID: 
        larp.group_id(p.part.group_id)
        
    if not p.part.interaction_id == supera.kINVALID_INSTANCEID:
        larp.interaction_id(p.part.interaction_id)
    
    return larp

def larcv_neutrino(n):
    
    larn = larcv.Neutrino()
    US2NS = 1.e3
        
    larn.id                  (larcv.InstanceID_t(n.idx)) 
    larn.interaction_id      (larcv.InstanceID_t(n.interaction_id))
    larn.current_type        (n.current_type)
    larn.interaction_mode    (n.interaction_mode)
    larn.interaction_type    (n.interaction_type)
    larn.target              (n.target)   
    larn.bjorken_x              (n.bjorken_x)
    larn.inelasticity           (n.inelasticity)
    larn.momentum_transfer      (n.momentum_transfer)
    larn.momentum_transfer_mag  (n.momentum_transfer_mag)
    larn.energy_transfer        (n.energy_transfer)
    larn.theta               (n.theta)
    larn.pdg_code            (int(n.pdg_code))
    larn.lepton_pdg_code     (int(n.lepton_pdg_code))
    larn.momentum            (n.px, n.py, n.pz)
    larn.lepton_p            (n.lepton_p)
    larn.position            (n.x, n.y, n.z, n.time * US2NS)
    larn.energy_init         (n.energy_init)
   
    return larn
  
def larcv_flash(f):

    larf=larcv.Flash()

    larf.id              (int(f.flash_id))
    larf.time            (f.time)
    larf.timeWidth       (f.timeWidth)
    larf.tpc             (f.tpc)
    pe_vec = cppyy.gbl.std.vector('double')() #direct conversion of vector doesn't seem to work
    for pe in (f.PEPerOpDet):
        pe_vec.push_back(pe)
    larf.PEPerOpDet      (pe_vec)

    return larf

def get_flow2supera(config_key):

    driver = flow2supera.driver.SuperaDriver()
    if os.path.isfile(config_key):
        driver.ConfigureFromFile(config_key)
    else:
        driver.ConfigureFromFile(flow2supera.config.get_config(config_key))
    

    return driver 

def log_supera_integrity_check(input_data, driver, log, verbose=False):

    if not log:
        return

    label = driver.Label()
    meta  = driver.Meta()

    # Packet tensor
    all_edeps = np.array([edep.e for edep in driver._edeps_all])
    all_voxel = meta.edep2voxelset(driver._edeps_all)
    all_voxel = np.array([vox.value() for vox in all_voxel.as_vector()])

    in_edeps = [[edep.e for edep in p.pcloud] for p in input_data if p.pcloud.size()]
    in_edeps_unass = np.array([edep.e for edep in driver._edeps_unassociated])
    in_voxel_unass = meta.edep2voxelset(driver._edeps_unassociated)
    in_voxel_unass = np.array([vox.value() for vox in in_voxel_unass.as_vector()])

    out_voxel_cluster = [[vox.value() for vox in p.energy.as_vector()] for p in label.Particles() if p.energy.size()]
    out_voxel_tensor  = np.array([vox.value() for vox in label._energies.as_vector()])
    out_voxel_unass   = np.array([vox.value() for vox in label._unassociated_voxels.as_vector()])

    if len(in_edeps): in_edeps = np.concatenate(in_edeps)
    if len(out_voxel_cluster): out_voxel_cluster = np.concatenate(out_voxel_cluster)

    log['raw_edeps_sum'].append(np.sum(all_edeps))
    log['raw_edeps_num'].append(len(all_edeps))
    log['raw_voxel_sum'].append(np.sum(all_voxel))
    log['raw_voxel_num'].append(len(all_voxel))

    log['in_edeps_sum'].append(np.sum(in_edeps))
    log['in_edeps_num'].append(len(in_edeps))
    log['in_edeps_unass_sum'].append(np.sum(in_edeps_unass))
    log['in_edeps_unass_num'].append(len(in_edeps_unass))
    log['in_voxel_unass_sum'].append(np.sum(in_voxel_unass))
    log['in_voxel_unass_num'].append(len(in_voxel_unass))
    
    log['out_voxel_tensor_sum'].append(np.sum(out_voxel_tensor))
    log['out_voxel_tensor_num'].append(len(out_voxel_tensor))
    log['out_voxel_cluster_sum'].append(np.sum(out_voxel_cluster))
    log['out_voxel_cluster_num'].append(len(out_voxel_cluster))
    log['out_voxel_unass_sum'].append(np.sum(out_voxel_unass))
    log['out_voxel_unass_num'].append(len(out_voxel_unass))

    
# Fill SuperaAtomic class and hand off to label-making
def run_supera(out_file='larcv.root',
               in_file='',
               config_key='',
               num_events=None,
               num_skip=0,
            #    ignore_bad_association=True,
               save_log=None,
               verbose=False):

    start_time = time.time()

    writer = get_iomanager(out_file)
  
    driver = get_flow2supera(config_key)

    reader = flow2supera.reader.InputReader(driver.parser_run_config(), config_key)

    if num_events is None:
        reader.ReadFile(in_file)
        num_events = len(reader)
    else:
        entries_to_read = int(num_events) + int(num_skip)
        reader.ReadFile(in_file, entries_to_read)

    id_vv = ROOT.std.vector("std::vector<unsigned long>")()
    value_vv = ROOT.std.vector("std::vector<float>")()

    id_v=ROOT.std.vector("unsigned long")()
    value_v=ROOT.std.vector("float")()

    if num_events < 0:
        num_events = len(reader)

    print("[run_supera] startup {:.3e} seconds".format(time.time() - start_time))

    LOG_KEYS  = ['event_id','time_read','time_convert','time_generate', 'time_store', 'time_event', 'time_log']
    LOG_KEYS += ['raw_edeps_sum','raw_edeps_num','raw_voxel_sum','raw_voxel_num',
    'in_edeps_sum','in_edeps_num','in_edeps_unass_sum','in_edeps_unass_num','in_voxel_unass_sum','in_voxel_unass_num',
    'out_voxel_tensor_sum','out_voxel_tensor_num','out_voxel_cluster_sum','out_voxel_cluster_num',
    'out_voxel_unass_sum','out_voxel_unass_num']

    logger = dict()
    if save_log:
        for key in LOG_KEYS:
            logger[key]=[]
        driver.log(logger)
    
    print(f"[run_supera] Processing {len(reader)} entries ")
    for entry in range(len(reader)):

        if num_skip and entry < num_skip:
            continue

        if num_events <= 0:
            break

        num_events -= 1 

        print(f'\n----- [run_supera] Processing Entry {entry} -----\n')

        t0 = time.time()
        input_data = reader.GetEntry(entry)
        #reader.EventDump(input_data)
        time_read = time.time() - t0
        print("[run_supera] reading input   {:.3e} seconds".format(time_read))
        if save_log: logger['time_read'].append(time_read)

        t1 = time.time()
        EventInput = driver.ReadEvent(input_data, is_sim=reader._is_sim)
        time_convert = time.time() - t1
        print("[run_supera] data conversion {:.3e} seconds".format(time_convert))
        if save_log: logger['time_convert'].append(time_convert)


        t2 = time.time()
        driver.GenerateImageMeta(EventInput)
            
        meta   = larcv_meta(driver.Meta())

        if not reader._is_sim:
            time_generate = time.time() - t2
            if save_log: logger['time_generate'].append(time_generate)

            t3 = time.time()
            tensor_hits = writer.get_data("sparse3d", "hits")
            driver.Meta().edep2voxelset(driver._edeps_unassociated).fill_std_vectors(id_v, value_v)
            larcv.as_event_sparse3d(tensor_hits, meta, id_v, value_v)          

        else:

            if input_data.trajectories is None:
                print(f'[run_supera] WARNING skipping this entry {entry} as it appears to be "empty" (no truth association found, non-unique event id, etc.)')
                continue

            driver.Meta().edep2voxelset(driver._edeps_all).fill_std_vectors(id_v, value_v)
            driver.GenerateLabel(EventInput) 
            time_generate = time.time() - t2
            print("[run_supera] label creation  {:.3e} seconds".format(time_generate))
            if save_log: logger['time_generate'].append(time_generate)

            # Start data store process
            t3 = time.time()

            result = driver.Label()
            tensor_energy = writer.get_data("sparse3d", "pcluster")
            result.FillTensorEnergy(id_v, value_v)
            larcv.as_event_sparse3d(tensor_energy, meta, id_v, value_v)
            
            # Check the input image and label image match in the voxel set
            #ids_input = np.array([v.id() for v in tensor_energy.as_vector()])
            #ids_label = np.array([v.id() for v in tensor_hits.as_vector()])
 
            #assert np.allclose(ids_input,ids_label), '[run_supera] ERROR: the label and input data has different set of voxels'

            tensor_semantic = writer.get_data("sparse3d", "pcluster_semantics")
            result.FillTensorSemantic(id_v, value_v)
            larcv.as_event_sparse3d(tensor_semantic,meta, id_v, value_v)

            cluster_energy = writer.get_data("cluster3d", "pcluster")
            result.FillClustersEnergy(id_vv, value_vv)
            larcv.as_event_cluster3d(cluster_energy, meta, id_vv, value_vv)

            cluster_dedx = writer.get_data("cluster3d", "pcluster_dedx")
            result.FillClustersdEdX(id_vv, value_vv)
            larcv.as_event_cluster3d(cluster_dedx, meta, id_vv, value_vv)

            particle = writer.get_data("particle", "pcluster")
            for p in result._particles:
                if not p.valid:
                    continue
                larp = larcv_particle(p)
                particle.append(larp)
            
            #Fill mc truth neutrino interactions
            interaction = writer.get_data("neutrino", "mc_truth")
            for ixn in input_data.interactions:
                if isinstance(ixn,np.void):
                    continue
                larn = larcv_neutrino(ixn)
                interaction.append(larn)

        #Fill flashes
        flash = writer.get_data("opflash", "light")
        for fl in input_data.flashes:
            larf = larcv_flash(fl)
            flash.append(larf)

        #propagating trigger info
        trigger = writer.get_data("trigger", "base")
        trigger.id(int(input_data.event_id))  # fixme: this will need to be different for real data?
        trigger.time_s(int(input_data.t0))
        trigger.time_ns(int(1e9 * (input_data.t0 - trigger.time_s())))

        # TODO fill the run ID 
        writer.set_id(0, 0, int(input_data.event_id))
        if save_log: logger['event_id'].append(input_data.event_id)
        writer.save_entry()

        time_store = time.time() - t3
        print("[run_supera] storing output  {:.3e} seconds".format(time_store))
        if save_log: logger['time_store'].append(time_store)

        # Perform an integrity check
        if save_log:
            t4 = time.time()
            log_supera_integrity_check(EventInput, driver, logger, verbose=False)
            time_log = time.time() - t4
            print("[run_supera] logging/check   {:.3e} seconds".format(time_log))
            if save_log: logger['time_log'].append(time_log)

        time_event = time.time() - t0
        print("[run_supera] driver total    {:.3e} seconds".format(time_event))
        if save_log: logger['time_event'].append(time_event)

    writer.finalize()

    
    if save_log:
        np.savez(save_log+'.npz',**logger)

    end_time = time.time()
    
    print("\n----- [run_suera] finished -----\n")
    print("[run_supera] Total processing time in s: ", end_time-start_time,'\n')








