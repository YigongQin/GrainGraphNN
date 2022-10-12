

class Updater(object):
    def __init__(self, iterable=(), **kwargs):
        self.__dict__.update(iterable, **kwargs)

class Param(Updater):
    def __init__(self, iterable=(), **kwargs):
       super().__init__(iterable, **kwargs)



def hyperparam(mode, all_id):


	'''
    user defined
	'''

	Ct = 1   
	Cl = 1   
	all_frames = 600*Ct + 1
	G = 32

	'''
    hyperparameter grid
	'''
	frames_pool=[20,24,30]
	learning_rate_pool=[25e-4, 50e-4, 100e-4]
	layers_pool=[3,4,5]
	hidden_dim_pool = [16, 24, 32]
	out_win_pool = [4, 5, 6] 

	frames_id = all_id//81
	lr_id = (all_id%81)//27
	layers_id = (all_id%27)//9
	hd_id = (all_id%9)//3
	owin_id = all_id%3


	frames = frames_pool[frames_id]*Ct+1
	learning_rate = learning_rate_pool[lr_id]
	layers = layers_pool[layers_id]
	hidden_dim=hidden_dim_pool[hd_id]
	out_win = out_win_pool[owin_id]


	if mode=='train' or mode=='test':
	  window = out_win
	  pred_frames = frames-window

	if mode=='ini':
	  learning_rate *= 2
	  window = 1
	  pred_frames = out_win - 1


	
	dt = Ct*1.0/(frames-1)




	param_dict = {'all_frames':all_frames, 'frames':frames, 'window':window, 'out_win':1, 'pred_frames':pred_frames, 'dt':dt, \
	             'layers':layers, 'layer_size':hidden_dim, 'kernel_size':(3,), 'lr':learning_rate, 'epoch':60, 'bias':True, 'model_list':[0], \
	             'Ct':Ct, 'Cl':Cl, 'G_base':8, 'G':G, \
	             'feature_dim':10, 'feat_list':['w','dw','s','y','w0','alpha','G','R','e_k','t']}


	return Param(param_dict)











