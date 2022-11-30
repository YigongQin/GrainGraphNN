

class Updater(object):
    def __init__(self, iterable=(), **kwargs):
        self.__dict__.update(iterable, **kwargs)

class Param(Updater):
    def __init__(self, iterable=(), **kwargs):
       super().__init__(iterable, **kwargs)



def regressor(mode, model_id):


    Ct = 1

    all_frames = 20*Ct + 1

        
    hp_grid = {'frames':[20, 24, 30], \
               'lr':[100e-4, 25e-4, 50e-4,  200e-4], \
               'layer':[2, 3, 4],\
               'hidden':[32, 24, 16],\
               'weight':[5, 10, 20, 40],\
               'decay_step':[10, 5, 20],\
               'batch_size':[2, 1, 4, 8]}
        
    
    hp_size = [(len(v),k) for k, v in hp_grid.items()]
    hp_order = [1, 5, 6, 3, 2, 0, 4]
    hp_size = [hp_size[i] for i in hp_order]

    param_dict = {}
    prev_dim = 1
    
    for grid_dim, param in hp_size:
        
        cur_dim = prev_dim*grid_dim
        
        
        param_idx = (model_id%cur_dim)//prev_dim
        param_dict.update({param:hp_grid[param][param_idx]})
        
        prev_dim = cur_dim
        

        
       
       
    param_dict['frames'] = param_dict['frames']*Ct+1



    out_win = 1
    if mode=='train' or mode=='test':
    	  window = out_win
    	  pred_frames = param_dict['frames']-window

    if mode=='ini':
    	  param_dict['lr'] *= 2
    	  window = 1
    	  pred_frames = out_win - 1


	
    dt = Ct*1.0/(param_dict['frames']-1)




    param_dict.update({'all_frames':all_frames, 'window':window, 'out_win':1, 'pred_frames':pred_frames, 'dt':dt, \
	             'layers':1, 'layer_size':32, 'kernel_size':(3,), 'epoch':60, 'bias':True, 'model_list':[0]})


    return Param(param_dict)




def classifier(mode, model_id):


        
    hp_grid = {'weight':[5, 10, 20, 40],\
               'lr':[100e-4, 25e-4, 50e-4,  200e-4], \
               'batch_size':[2, 1, 4, 8],\
               'decay_step':[10, 5, 20],\
               'hidden':[32, 24, 16]}
        
    
    hp_size = [(len(v),k) for k, v in hp_grid.items()]
    hp_order = [0, 1, 2, 3, 4]
    hp_size = [hp_size[i] for i in hp_order]

    param_dict = {}
    prev_dim = 1
    
    for grid_dim, param in hp_size:
        
        cur_dim = prev_dim*grid_dim
        
        
        param_idx = (model_id%cur_dim)//prev_dim
        param_dict.update({param:hp_grid[param][param_idx]})
        
        prev_dim = cur_dim
 
       
       
    param_dict['frames'] = 13

    param_dict.update({'window':1, 'out_win':1,\
	             'layers':1, 'layer_size':32, 'kernel_size':(3,), 'epoch':60, 'bias':True, 'model_list':[0]})


    return Param(param_dict)




"""
frames_id = all_id//81
lr_id = (all_id%81)//27
layers_id = (all_id%27)//9
hd_id = (all_id%9)//3
owin_id = all_id%3
"""


