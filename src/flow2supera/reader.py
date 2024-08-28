import h5py 
import h5flow
import numpy as np
from ROOT import supera
from yaml import Loader
import yaml
import os
import flow2supera
import time
import tqdm


class InputEvent:
    event_id = -1
    true_event_id = -1
    segments = None
    hit_indices = None
    hits = None
    backtracked_hits = None
    calib_final_hits  = None
    trajectories = None
    interactions = []
    t0 = -1
    segment_index_min = -1
    event_separator = ''


class InputReader:
    
    def __init__(self, parser_run_config, input_files=None,config=None):
        self._input_files = input_files
        if not isinstance(input_files, str):
            raise TypeError('Input file must be a str type')
        self._event_ids = None
        self._event_t0s = None
        self._flash_t0s = None
        self._flash_ids = None
        self._event_hit_indices = None
        self._hits = None
        self._backtracked_hits = None
        self._segments = None
        self._trajectories = None
        self._interactions = None
        self._run_config = parser_run_config
        self._is_sim = False
        self._is_mpvmpr= False
        if config:
            if os.path.isfile(config):
                file=config
            else:
                file=flow2supera.config.get_config(config)
            with open(file,'r') as f:
                cfg=yaml.safe_load(f.read())
                if 'Flow2Supera' in cfg.keys():
                    if not isinstance(cfg['Flow2Supera'],dict):
                        raise TypeError('Flow2Supera configuration block should be a dict type')

                    if 'DataType' in cfg['Flow2Supera']:
                        self._is_sim=cfg['Flow2Supera'].get('DataType')[0]=='sim'
                        self._is_mpvmpr=cfg['Flow2Supera'].get('DataType')[1]=='mpvmpr'  
                
        print(f'[InputReader] is sim? {self._is_sim} is mpvmpr? {self._is_mpvmpr}')

        
        if input_files:
            self.ReadFile(input_files)

    def __len__(self):
        if self._event_ids is None: return 0
        return len(self._event_ids)

    
    def __iter__(self):
        for entry in range(len(self)):
            yield self.GetEvent(entry)

    def ReadFile(self, input_files, verbose=False):
        
        print('Reading input file...')

        # H5Flow's H5FlowDataManager class associated datasets through references
        # These paths help us get the correct associations

        events_path            = 'charge/events/'
        events_data_path       = 'charge/events/data/'
        event_hit_indices_path = 'charge/events/ref/charge/calib_prompt_hits/ref_region/'
        packets_path           = 'charge/packets'
        calib_final_hits_path  = 'charge/calib_final_hits/data'
        calib_prompt_hits_path = 'charge/calib_prompt_hits/data'

        backtracked_hits_path  = 'mc_truth/calib_prompt_hit_backtrack/data'
        interactions_path      = 'mc_truth/interactions/data'
        segments_path          = 'mc_truth/segments/data'
        trajectories_path      = 'mc_truth/trajectories/data'
        
        # TODO Currently only reading one input file at a time. Is it 
        # necessary to read multiple? If so, how to handle non-unique
        # event IDs?
        #for f in input_files:
        flow_manager = h5flow.data.H5FlowDataManager(input_files, 'r')
        with h5py.File(input_files, 'r') as fin:
            events = flow_manager[events_path]
            events_data = events['data']
            self._event_ids = events_data['id']
            #ts_start is in ticks and 0.1 microseconds per tick for charge readout
            self._event_t0s = events_data['unix_ts'] + events_data['ts_start']/1e7 
            self._event_hit_indices = flow_manager[event_hit_indices_path]
            self._hits              = flow_manager[calib_prompt_hits_path]
            #self._is_sim = 'mc_truth' in fin.keys()
            if self._is_sim:
                self._backtracked_hits  = flow_manager[backtracked_hits_path]
                self._segments     = np.array(flow_manager[segments_path])
                self._trajectories = np.array(flow_manager[trajectories_path])
                self._interactions = np.array(flow_manager[interactions_path])
                
                # Make explicit reference to segment ids and entry index array
                self._segment_ids = self._segments['segment_id']
                self._segment_idx = np.arange(len(self._segments))
                self._segment_event_ids = self._segments['event_id']

                # Quality check: event IDs from segments are consistent with the info stored at the event level
                if not len(self._event_hit_indices) == len(self._event_ids):
                    print('The number of entries do not match between event_data and backtrack hit range array')
                    print(event_path,'...',len(self._event_ids))
                    print(event_hit_indices_path,'...',len(self._event_hit_indices))
                    raise ValueError('Array length mismatch in the input file')
                self._valid_segment_event_ids = self.FileQualityCheck()

    
    def GetNeutrinoIxn(self, ixn, ixn_idx):
        
        nu_result = supera.Neutrino()

        if isinstance(ixn,np.void):
            return nu_result
        
        nu_result.id = int(ixn_idx)
        nu_result.interaction_id = int(ixn['vertex_id']) 
        nu_result.target = int(ixn['target'])
        nu_result.vtx = supera.Vertex(ixn['x_vert'], ixn['y_vert'], ixn['z_vert'], ixn['t_vert'])
        nu_result.pdg_code = int(ixn['nu_pdg'])
        nu_result.lepton_pdg_code = int(ixn['lep_pdg'])  
        nu_result.energy_init = ixn['Enu']
        nu_result.theta = ixn['lep_ang']
        nu_result.momentum_transfer =  ixn['Q2']
        nu_result.momentum_transfer_mag =  ixn['q3']
        nu_result.energy_transfer =  ixn['q0']
        nu_result.bjorken_x = ixn['x']
        nu_result.inelasticity = ixn['y']
        nu_result.px = ixn['nu_4mom'][0]
        nu_result.py = ixn['nu_4mom'][1]       
        nu_result.pz = ixn['nu_4mom'][2]
        nu_result.lepton_p = ixn['lep_mom']
        if(ixn['isCC']): nu_result.current_type = 0
        else: nu_result.current_type = 1
        nu_result.interaction_mode = int(ixn['reaction'])
        nu_result.interaction_type = int(ixn['reaction'])   
        
        return nu_result  
        
    # To truth associations go as hits -> segments -> trajectories


    def GetEventIDFromSegments(self, backtracked_hits):
        
        try:

            seg_ids = np.unique(np.concatenate([bhit['segment_ids'][bhit['fraction']!=0.] for bhit in backtracked_hits]))

            sid_min,sid_max = seg_ids.min(),seg_ids.max()

            seg_range_mask = (self._segment_ids >= sid_min) & (self._segment_ids <= sid_max)

            event_segs=self._segment_ids[seg_range_mask]
            event_idxs=self._segment_idx[seg_range_mask]

            seg_mask = [event_idxs[i] for i in range(len(event_segs)) if event_segs[i] in seg_ids]

            return np.unique(self._segment_event_ids[seg_mask])

        except ValueError:
            valid_frac_counts = [(bhit['fraction']!=0.).sum() for bhit in backtracked_hits]
            if sum(valid_frac_counts) > 0:
                # case the original error was not due to empty association, re-raise
                raise
            print(f'[InputReader] UNEXPECTED: found no hit with any association to the truth hit')
            return np.array([])


    def FileQualityCheck(self):

        eid_ctr = np.zeros(len(self._event_hit_indices),dtype=int)
        eid_val = np.full(len(self._event_hit_indices),fill_value=-1,dtype=int)
        bad_event_ids = []
        empty_entries = []
        
        print('[InputReader] Checking the event IDs in this file...')
        for entry,(hidx_min,hidx_max) in tqdm.tqdm(enumerate(self._event_hit_indices),desc='Scanning event IDs'):
            bhits = self._backtracked_hits[hidx_min:hidx_max]
            if len(bhits) == 0: 
                empty_entries.append(entry)
                continue
            ids_this=self.GetEventIDFromSegments(bhits)
            if not len(ids_this) == 1:
                eid_ctr[entry] = len(ids_this)
                if len(ids_this):
                    bad_event_ids.append(ids_this)
            else:
                eid_val[entry] = ids_this[0]

        bad_entries = (eid_val == -1).nonzero()[0]
        
        # Filter out bad entries that are in empty_entries
        bad_entries = [entry for entry in bad_entries if entry not in empty_entries]
        
        if len(empty_entries) > 0:
            print('[Inputreader] WARNING: These entries have no hits:', empty_entries)
 
        if len(bad_entries) > 0:
            print('[Inputreader] WARNING: entries where more than one event ID is found:',bad_entries)
            print('                       corresponding event IDs stored:',[list(ids) for ids in bad_event_ids])

        # Find other impacted entries
        if len(bad_event_ids):
            bad_event_ids=np.concatenate(bad_event_ids)
        bad_event_ids = np.unique(bad_event_ids)
        mask=np.zeros(len(self._event_hit_indices),dtype=bool)
        for bad_id in bad_event_ids:
            mask = mask | (eid_val == bad_id)
        if mask.sum():
            print('[Inputreader] WARNING: other entries impacted by bad event IDs:',mask.nonzero()[0])

        entry_mask = mask | (eid_val == -1)
        eid_val[entry_mask] = -1

        # if bad_entries:
        #     raise ValueError("ERROR: Terminating due to multiple true event id association. Check simulation")
            
        return eid_val
        

    def GetEntry(self, entry):
        
        if entry >= len(self._event_ids):
            print('[Inputreader] Entry {} is above allowed entry index ({})'.format(entry, len(self._event_ids)))
            print('              Invalid read request (returning None)')
            return None

        t0=time.time()
        result = InputEvent()

        result.event_id = self._event_ids[entry]

        result.t0 = self._event_t0s[entry] 

        result.hit_indices = self._event_hit_indices[entry]
        hidx_min, hidx_max = self._event_hit_indices[entry]
        result.hits = self._hits[hidx_min:hidx_max]
        
        if not self._is_sim:
            print('[InputReader] SuperaInput filled (not sim)',time.time()-t0,'[s]')
            return result
        
        result.backtracked_hits = self._backtracked_hits[hidx_min:hidx_max]
        
        if self._valid_segment_event_ids[entry] < 0:
            print(f'[InputReader] Skipping this entry ({entry})...')
            return result
        
        st_event_id = self._valid_segment_event_ids[entry]


        result.segments = self._segments[self._segments['event_id']==st_event_id]
        result.trajectories = self._trajectories[self._trajectories['event_id']==st_event_id]
        
        if self._is_mpvmpr:
            print('[InputReader] SuperaInput filled (sim, mpvmpr)',time.time()-t0,'[s]')
            return result
        
        result.interactions = []
        
        result.true_event_id = st_event_id      
        interactions_array  = np.array(self._interactions)
        event_interactions = interactions_array[interactions_array['event_id'] == result.true_event_id]
        for ixn_idx, ixn in enumerate(event_interactions):
            supera_nu = self.GetNeutrinoIxn(ixn, ixn_idx)
            result.interactions.append(supera_nu)  
        
        print('[InputReader] SuperaInput filled (sim, not mpvmpr)',time.time()-t0,'[s]')
        return result 


    def EventDump(self, input_event):
        print('-----------EVENT DUMP [InputReader] ------------')
        print('Event ID {}'.format(input_event.event_id))
        print('Event t0 {}'.format(input_event.t0))
        print('Event hit indices (start, stop):', input_event.hit_indices)
        print('hits shape:', input_event.hits.shape)
        if self._is_sim and len(input_event.hits) !=0:
            print('True event ID {}'.format(input_event.true_event_id))
            print('Backtracked hits len:', len(input_event.backtracked_hits))
            print('segments in this event:', len(input_event.segments))
            print('trajectories in this event:', len(input_event.trajectories))
            print('interactions in this event:', len(input_event.interactions))


