
# import Jump_SSM_2 as kf
# import PyKalman_29_3blocks_1 as EM_kf
#import PyKalman as kf
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt===
# from joblib import Parallel, delayed
import datetime
# from pylab import rcParams
# from ggplot import *
from scipy.optimize import minimize
# import matplotlib.pyplot as plt
import statsmodels.api as sm
# from joblib import Parallel, delayed
from scipy import optimize
from timeit import default_timer as timer
import Maximzers_jumps
import Maximzers
import multiprocessing
import ctypes
import sys
import math


#===============================================================================================--
indicator = int(sys.argv[1])																
str_indicator = str(indicator)
permno=pd.read_csv("/scratch/khansaad/2005/CC_permno_1500_2005_"+str_indicator+".csv")

																								#(1)			(2)			(3)			(4)			(5)				(6)				(7)			(8)			(9)				(10)		(11)		(12)	(13)		(14)
series=pd.read_csv("/scratch/khansaad/2005/CC_SP_1500_2005_jumps_"+str_indicator+".csv",usecols=["date_time_m", "PERMNO", "OVER_COUNTS","COUNT_JUMPS","YEAR_MONTH","date_q","orderflow_5","orderflow_5_inno","indicators","log_midpoint","OVER_RET","RET","JUMP_5P_ret","OVER"])

#==================================================================================================--






series['date_time_m_2']=pd.to_datetime(series['date_time_m'], format='%d%b%Y:%H:%M:%S.%f')











		#==========================================================================================
		#					PARAMETERS

def scanloop(permno,date_q, original_order):

	#CREATING LABEL FOR FILE
	label_1=str(permno)
	label_2=str(date_q)
	seperator="_"
	label = label_1 + seperator + label_2

	print("label is ", label)

	orig_stdout = sys.stdout
	f = open("/home/khansaad/SP1500/2005/Results/Logs/permno_%s.txt"%(label), 'w')
	sys.stdout = f

	estimates_mle = np.zeros(shape=(1, 60)) 



	start = timer()


	#=====================================
	
	series_2=series[(series['PERMNO']==permno) & (series['date_q']==date_q)]	

	series_2=series_2.sort_values(by='date_time_m_2', ascending=True)
	series_2_orig = series_2
	#============================================

	if series_2['OVER_COUNTS'].iloc[1]<3 or series_2['COUNT_JUMPS'].iloc[1]<3:
		print("counts too low",series_2['OVER_COUNTS'].iloc[0],series_2['COUNT_JUMPS'].iloc[0])
		estimates_mle[0,0] = None
		estimates_mle[0,1] = None
		estimates_mle[0,2] = None
		estimates_mle[0,3] = None
		estimates_mle[0,4] = None
		estimates_mle[0,5] = None
		estimates_mle[0,6] = None
		estimates_mle[0,7] = None
		estimates_mle[0,8] = None
		estimates_mle[0,9] = None 
		estimates_mle[0,10] = None 
		estimates_mle[0,11] = None
		estimates_mle[0,12] = None
		estimates_mle[0,13] = None
		estimates_mle[0,14] = None
		estimates_mle[0,15] = None
		estimates_mle[0,16] = None
		estimates_mle[0,17] = None
		estimates_mle[0,18] = None
		estimates_mle[0,19] = None
		estimates_mle[0,20] = None
		estimates_mle[0,21] = permno
		estimates_mle[0,22] = series_2['YEAR_MONTH'].iloc[0]
		estimates_mle[0,23] = None
		estimates_mle[0,24] = None
		estimates_mle[0,25] = None
		estimates_mle[0,26] = None
		estimates_mle[0,27] = None
		estimates_mle[0,28] = None
		estimates_mle[0,29] = None
		estimates_mle[0,30] = None 
		estimates_mle[0,31] = None
		estimates_mle[0,32] = None
		estimates_mle[0,33] = None
		estimates_mle[0,34] = None
		estimates_mle[0,35] = None
		estimates_mle[0,36] = None
		estimates_mle[0,37] = None
		estimates_mle[0,38] =None
		estimates_mle[0,39] =None
		estimates_mle[0,40] =None
		estimates_mle[0,41] =None
		estimates_mle[0,42] =None
		estimates_mle[0,43] =None
		estimates_mle[0,44] =None
		estimates_mle[0,45] =None
		estimates_mle[0,46] =None
		estimates_mle[0,47] =series_2['COUNT_JUMPS'].iloc[1]
		estimates_mle[0,48] =None
		estimates_mle[0,49] =series_2['OVER_COUNTS'].iloc[1]
		estimates_mle[0,50]=np.std(series_2['orderflow_5'].values)
		estimates_mle[0,51]=np.std(series_2['orderflow_5_inno'].values)
		estimates_mle[0,52] = None
		estimates_mle[0,53]=  None
		estimates_mle[0,54]=  None
		estimates_mle[0,55]= None
		estimates_mle[0,56]= 0
		estimates_mle[0,57]= 0
		estimates_mle[0,58]= series_2['YEAR_MONTH'].iloc[-1]
		estimates_mle[0,59]= 1

		print("label is ", label)

		np.savetxt("/home/khansaad/SP1500/2005/Results/permno_%s.csv"%(label),estimates_mle, delimiter=",")

		return
	#============================================


	print(series_2_orig)


	
	indicators = series_2_orig['indicators'].values


	initial_state_m=series_2['log_midpoint'].iloc[0]
	series_2=series_2.iloc[1:]
	series_2=series_2.sort_values(by='date_time_m_2', ascending=True)



	orderflow_inno=series_2['orderflow_5_inno'].values/np.std(series_2['orderflow_5_inno'].values)
	orderflow_fit=series_2['orderflow_5'].values/np.std(series_2['orderflow_5'].values)

	series_jump = series_2[series_2['JUMP_5P_ret']==1]
	series_over = series_2[series_2['OVER']==1]
	over_var = np.var(series_over["OVER_RET"].values)
	jump_var = np.var(series_jump["RET"].values)

	
	#============================================
	if jump_var==None or over_var==None:
		print("jump/over variances were ",jump_var,over_var)
		return
	if math.isnan(jump_var)==True or math.isnan(over_var)==True:
		print("jump/over variances were ",jump_var,over_var)
		return
	#============================================


		
	print(series_2[series_2['OVER']==1])
	print(series_over["OVER_RET"].values)

	#JUMPS HAVE ORDERFLOW KEEP JUMPS
	series_cont = series_2[(series_2['OVER']==0)]
	series_cont=series_cont.sort_values(by='date_time_m_2', ascending=True)

	orderflow_inno_cont=series_cont['orderflow_5_inno'].values/np.std(series_cont['orderflow_5_inno'].values)
	orderflow_fit_cont=series_cont['orderflow_5'].values/np.std(series_cont['orderflow_5'].values)


	print("jump and over ret variances are ", jump_var, over_var)

#===============================================================================
	kalman=EM_kf.KalmanFilter( observation_matrices=np.matrix([1,1]), 
	initial_state_mean=[initial_state_m,0], observation_covariance=0,transition_matrices=np.matrix([[1,0],[0,0]]),
	random_state=None,
	n_dim_state=2, n_dim_obs=1)

	em = kalman.em(X=series_cont.log_midpoint.values, orderflow_inno= orderflow_inno_cont, orderflow_fit=orderflow_fit_cont,em_vars=['transition_covariance','initial_state_covariance','kappa','beta'],n_iter=100)

	cont_perm_var = kalman.transition_covariance[0,0]
	cont_trans_var = kalman.transition_covariance[1,1]

#===============================================================================
#===============================================================================
	Maximzers_defined_cont = Maximzers.Implement(observations=series_cont.log_midpoint.values, orderflow_inno=orderflow_inno_cont, orderflow_fit=orderflow_fit_cont, initial_state_m=initial_state_m)
		
		#Si Brute - Cont
	si= Maximzers_defined_cont.Brute_maximizer_si(perm_var=kalman.transition_covariance[0,0], trans_var=kalman.transition_covariance[1,1], 
	kappa=kalman.kappa, beta=kalman.beta, si_range_start=-0.7,si_range_end=0.7,step=0.01)
		#Si Contrained - Cont
	
	norm= np.absolute(kalman.beta/kalman.kappa)
	print("norm is ", norm)


	constrained_mle_cont= Maximzers_defined_cont.Constraint_MLE_maximizer( perm_var=kalman.transition_covariance[0,0], trans_var=kalman.transition_covariance[1,1], 
	kappa=kalman.kappa*norm, beta=kalman.beta, si=si,si_bnd_start=-0.7, si_bnd_end=0.7, gtol=1e-3, ftol=1e-5,norm=norm)
	print("continous si contrained is", constrained_mle_cont[4])


	print(" ")
	print("continous mle contrained etimators are ", np.exp(constrained_mle_cont[0]),np.exp(constrained_mle_cont[1]),
	constrained_mle_cont[2],constrained_mle_cont[3],constrained_mle_cont[4])
# #===============================================================================
	





	Maximzers_defined = Maximzers_jumps.Implement(observations=series_2.log_midpoint.values, indicators= indicators,orderflow_inno=orderflow_inno, orderflow_fit=orderflow_fit, initial_state_m=initial_state_m)
# 		#===========================================================================
# 		#LOG TRANSFORMATION IS TAKEN WITHIN THE MAXIMIZER ROUTINE - ENTER LEVELS
# 		#OUTPUTS TO CONSTRAINTS AND UNCONSTRAINT MAXIMIZER IS IN LOGS
# 		#============================================================================

	brute= Maximzers_defined.Brute_maximizer(perm_var_cont=np.exp(constrained_mle_cont[0]), trans_var_cont=np.exp(constrained_mle_cont[1]),
	kappa=(constrained_mle_cont[2]), beta=(constrained_mle_cont[3]),si=(constrained_mle_cont[4]),
	perm_var_jump_range_start=jump_var*(1/2),perm_var_jump_range_end=jump_var,perm_var_jump_step=jump_var/10,
	trans_var_jump_range_start=jump_var*(1/200),trans_var_jump_range_end=jump_var/2,trans_var_jump_step=jump_var/10,
	perm_var_over_range_start=over_var*(1/2),perm_var_over_range_end=over_var,perm_var_over_step=over_var/10,
	trans_var_over_range_start=over_var*(1/200),trans_var_over_range_end=over_var/2,trans_var_over_step=over_var/10)

	# print(" ")
	print("brute variances are", brute)
	print(" ")
	print("continous mle contrained etimators are ", np.exp(constrained_mle_cont[0]),np.exp(constrained_mle_cont[1]),
		constrained_mle_cont[2],constrained_mle_cont[3],constrained_mle_cont[4])



	norm= np.absolute(constrained_mle_cont[3]/constrained_mle_cont[2])
	print("norm is ", norm)

	(_,constrained_mle_1)= Maximzers_defined.Constraint_MLE_maximizer(
		perm_var_cont=np.exp(constrained_mle_cont[0]), 
		trans_var_cont=np.exp(constrained_mle_cont[1]),
		kappa=constrained_mle_cont[2]*norm, beta=constrained_mle_cont[3], si=constrained_mle_cont[4],
		perm_var_jump=brute[0], 
		trans_var_jump=brute[1],
		perm_var_overnight=brute[2],
		trans_var_overnight=brute[3],
		gtol=1e-1, ftol=1e-5,si_bnd_start=(si-0.15), si_bnd_end=(si+0.15),norm=norm)

	norm= np.absolute(constrained_mle_1[7]/constrained_mle_1[6])
	print("norm is ", norm)

	(_,constrained_mle_2)= Maximzers_defined.Constraint_MLE_maximizer(
		perm_var_cont=np.exp(constrained_mle_1[0]), 
		trans_var_cont=np.exp(constrained_mle_1[1]),
		perm_var_jump=np.exp(constrained_mle_1[2]), 
		trans_var_jump=np.exp(constrained_mle_1[3]),
		perm_var_overnight=np.exp(constrained_mle_1[4]),
		trans_var_overnight=np.exp(constrained_mle_1[5]),
		kappa=constrained_mle_1[6]*norm, beta=constrained_mle_1[7], si=constrained_mle_1[8],
	gtol=1e-2, ftol=1e-5, si_bnd_start=-0.8, si_bnd_end=0.8, norm=norm)


	print("contraint estimates are ", np.exp(constrained_mle_2[0]), np.exp(constrained_mle_2[1]),
		np.exp(constrained_mle_2[2]), np.exp(constrained_mle_2[3]), np.exp(constrained_mle_2[4]),
			np.exp(constrained_mle_2[5]),constrained_mle_2[6],constrained_mle_2[7],constrained_mle_2[8])

	norm_1= np.absolute(constrained_mle_2[7]/constrained_mle_2[6])
	print("norm is ", norm_1)

	(message_2,hessian_inv_3,fun_2,constrained_mle_3)=Maximzers_defined.Constraint_MLE_maximizer_si_fixed(
		perm_var_cont=np.exp(constrained_mle_2[0]), 
		trans_var_cont=np.exp(constrained_mle_2[1]), 
		perm_var_jump=np.exp(constrained_mle_2[2]), 
		trans_var_jump=np.exp(constrained_mle_2[3]),
		perm_var_overnight=np.exp(constrained_mle_2[4]), 
		trans_var_overnight=np.exp(constrained_mle_2[5]),
		kappa=constrained_mle_2[6]*norm_1, beta=constrained_mle_2[7], si=constrained_mle_2[8],gtol=5e-1,norm=norm_1)

	if  fun_2 != -1000000000000 and message_2=="Optimization terminated successfully." :
		initial_message=1
		hessian_inv_3 = hessian_inv_3*np.identity(8)
		np.savetxt("/home/khansaad/SP1500/2005/Results/Hessians/HessianInv_3_permno_%s.csv"%(label), hessian_inv_3, delimiter=",")
		np.savetxt("/home/khansaad/SP1500/2005/Results/Cons/Cons_permno_%s.csv"%(label), constrained_mle_3, delimiter=",")



	else:
		initial_message=0

	norm_2= np.absolute(constrained_mle_3[7]/ constrained_mle_3[6])

	print("norm is ", norm_2)

	(unconstrained_message,hessian_inv_2,fun, unconstrained_mle)= Maximzers_defined.Unconstraint_MLE_maximizer(
		perm_var_cont=np.exp(constrained_mle_3[0]), 
		trans_var_cont=np.exp(constrained_mle_3[1]), 
		perm_var_jump=np.exp(constrained_mle_3[2]), 
		trans_var_jump=np.exp(constrained_mle_3[3]),
		perm_var_overnight=np.exp(constrained_mle_3[4]), 
		trans_var_overnight=np.exp(constrained_mle_3[5]),
		kappa=constrained_mle_3[6]*norm_2, beta=constrained_mle_3[7], si=constrained_mle_2[8],gtol=5e-1,norm=norm_2)

	hessian_inv_2 = hessian_inv_2*np.identity(9)
	#IF ALL ELEMENTS ARE IDENTITY IT MEANS THE ALGORITHM WAS AREADY AT MAXIMUM===

	if (hessian_inv_2==np.identity(9)).all()==True:
		hessian_inv_2[0:8,0:8]=hessian_inv_3
	else:
		hessian_inv_2=hessian_inv_2


	print("label is ", label)
	#========================================================================================= 
	np.savetxt("/home/khansaad/SP1500/2005/Results/Hessians/HessianInv_2_permno_%s.csv"%(label), hessian_inv_2, delimiter=",")
	np.savetxt("/home/khansaad/SP1500/2005/Results/UnCons/Uncons_permno_%s.csv"%(label), unconstrained_mle, delimiter=",")





	print(" ")
	print("uncontrained_mle are ", np.exp(unconstrained_mle[0]),np.exp(unconstrained_mle[1]),np.exp(unconstrained_mle[2]),np.exp(unconstrained_mle[3]),
	np.exp(unconstrained_mle[4]),np.exp(unconstrained_mle[5]),unconstrained_mle[6],unconstrained_mle[7],unconstrained_mle[8])

	end = timer()
	time_taken = (end - start)/60
	print("time taken is" ,time_taken)


	if (fun != -1000000000000) and (unconstrained_message=="Optimization terminated successfully." or  message_2=="Optimization terminated successfully."):

		if unconstrained_message=="Optimization terminated successfully.":

			(output_mle) = unconstrained_mle

			perm_var_cont_mle = np.exp(output_mle[0])
			trans_var_cont_mle = np.exp(output_mle[1])
			perm_var_jump_mle = np.exp(output_mle[2])
			trans_var_jump_mle = np.exp(output_mle[3])
			perm_var_overnight_mle = np.exp(output_mle[4])
			trans_var_overnight_mle = np.exp(output_mle[5])
			#ONLY KAPPA IS NORMALIZED
			kappa_mle = output_mle[6]/norm_2
			beta_mle = output_mle[7]
			si_mle = output_mle[8]

		elif message_2=="Optimization terminated successfully.":
			(output_mle) = constrained_mle_3


			perm_var_cont_mle = np.exp(output_mle[0])
			trans_var_cont_mle = np.exp(output_mle[1])
			perm_var_jump_mle = np.exp(output_mle[2])
			trans_var_jump_mle = np.exp(output_mle[3])
			perm_var_overnight_mle = np.exp(output_mle[4])
			trans_var_overnight_mle = np.exp(output_mle[5])
			#ONLY KAPPA IS NORMALIZED
			kappa_mle = output_mle[6]/norm_1
			beta_mle = output_mle[7]
			si_mle = constrained_mle_2[8]
			hessian_inv_2[0:8,0:8]=hessian_inv_3


		hessian=Maximzers_defined.Compute_Hessian(perm_var_cont= perm_var_cont_mle,trans_var_cont=trans_var_cont_mle, 
		perm_var_jump= perm_var_jump_mle,trans_var_jump=trans_var_jump_mle, 
		perm_var_overnight= perm_var_overnight_mle,trans_var_overnight=trans_var_overnight_mle, 	
		kappa= kappa_mle, beta= beta_mle, si=si_mle)

		if Maximzers_defined.is_pos_def(hessian)!=False:
			hessian_inv = np.linalg.inv(hessian)
			hessian_inv=np.reshape(hessian_inv,(9,9))

			print("label is ", label)
			#======================================================================================== 
			np.savetxt("/home/khansaad/SP1500/2005/Results/Hessians/HessianInv_permno_%s.csv"%(label), hessian_inv, delimiter=",")
				#T-STATISTICS
			std_perm_var_cont_mle = np.sqrt(np.exp(output_mle[0])*(hessian_inv[0,0])*np.exp(output_mle[0]))
			std_trans_var_cont_mle = np.sqrt(np.exp(output_mle[1])*(hessian_inv[1,1])*np.exp(output_mle[1]))
			std_perm_var_jump_mle = np.sqrt(np.exp(output_mle[2])*(hessian_inv[2,2]*np.exp(output_mle[2])))
			std_trans_var_jump_mle =np.sqrt(np.exp(output_mle[3])*(hessian_inv[3,3])*np.exp(output_mle[3]))
			std_perm_var_overnight_mle =np.sqrt(np.exp(output_mle[4])*(hessian_inv[4,4])*np.exp(output_mle[4]))
			std_trans_var_overnight_mle = np.sqrt(np.exp(output_mle[5])*(hessian_inv[5,5])*np.exp(output_mle[5]))

			#=================================
			std_kappa_mle = np.sqrt(hessian_inv[6,6])
			std_beta_mle =np.sqrt(hessian_inv[7,7])
			std_si_mle = np.sqrt(hessian_inv[8,8])

			T_perm_var_cont_mle = perm_var_cont_mle/std_perm_var_cont_mle
			T_trans_var_cont_mle = trans_var_cont_mle/std_trans_var_cont_mle
			T_perm_var_jump_mle = perm_var_jump_mle/std_perm_var_jump_mle
			T_trans_var_jump_mle = trans_var_jump_mle/std_trans_var_jump_mle
			T_perm_var_overnight_mle = perm_var_overnight_mle/std_perm_var_overnight_mle
			T_trans_var_overnight_mle = trans_var_overnight_mle/std_trans_var_overnight_mle
			T_kappa_mle = kappa_mle/std_kappa_mle
			T_beta_mle = beta_mle/std_beta_mle
			T_si_mle = si_mle/std_si_mle
		else:
			#T-STATISTICS
			std_perm_var_cont_mle = None
			std_trans_var_cont_mle = None
			std_perm_var_jump_mle = None
			std_trans_var_jump_mle =None
			std_perm_var_overnight_mle =None
			std_trans_var_overnight_mle = None

			#=================================
			std_kappa_mle = None
			std_beta_mle =None
			std_si_mle = None

			T_perm_var_cont_mle = None
			T_trans_var_cont_mle = None
			T_perm_var_jump_mle = None
			T_trans_var_jump_mle = None
			T_perm_var_overnight_mle = None
			T_trans_var_overnight_mle = None
			T_kappa_mle = None
			T_beta_mle = None
			T_si_mle = None


		std_perm_var_cont_2 = np.sqrt(np.exp(output_mle[0])*(hessian_inv_2[0,0])*np.exp(output_mle[0]))
		std_trans_var_cont_2 = np.sqrt(np.exp(output_mle[1])*(hessian_inv_2[1,1])*np.exp(output_mle[1]))
		std_perm_var_jump_2 = np.sqrt(np.exp(output_mle[2])*(hessian_inv_2[2,2]*np.exp(output_mle[2])))
		std_trans_var_jump_2 =np.sqrt(np.exp(output_mle[3])*(hessian_inv_2[3,3])*np.exp(output_mle[3]))
		std_perm_var_overnight_2 =np.sqrt(np.exp(output_mle[4])*(hessian_inv_2[4,4])*np.exp(output_mle[4]))
		std_trans_var_overnight_2 = np.sqrt(np.exp(output_mle[5])*(hessian_inv_2[5,5])*np.exp(output_mle[5]))

		#=================================
		std_kappa_2 = np.sqrt(hessian_inv_2[6,6])
		std_beta_2 =np.sqrt(hessian_inv_2[7,7])
		std_si_2 = np.sqrt(hessian_inv_2[8,8])



		T_perm_var_cont_2 = perm_var_cont_mle/std_perm_var_cont_2
		T_trans_var_cont_2 = trans_var_cont_mle/std_trans_var_cont_2
		T_perm_var_jump_2 = perm_var_jump_mle/std_perm_var_jump_2
		T_trans_var_jump_2 = trans_var_jump_mle/std_trans_var_jump_2
		T_perm_var_overnight_2 = perm_var_overnight_mle/std_perm_var_overnight_2
		T_trans_var_overnight_2 = trans_var_overnight_mle/std_trans_var_overnight_2
		T_kappa_2 = kappa_mle/std_kappa_2
		T_beta_2 = beta_mle/std_beta_2
		T_si_2 = si_mle/std_si_2






		if unconstrained_message=="Optimization terminated successfully.":
			estimates_mle[0,0] = 1
		else:
			estimates_mle[0,0] = 2
		end = timer()
		time_taken = (end - start)/60

		estimates_mle[0,1] = perm_var_cont_mle 
		estimates_mle[0,2] = trans_var_cont_mle 
		estimates_mle[0,3] = perm_var_jump_mle 
		estimates_mle[0,4] = trans_var_jump_mle
		estimates_mle[0,5] = perm_var_overnight_mle
		estimates_mle[0,6] = trans_var_overnight_mle 
		estimates_mle[0,7] = kappa_mle 
		estimates_mle[0,8] = beta_mle
		estimates_mle[0,9] = si_mle 
		estimates_mle[0,10] = std_perm_var_cont_mle 
		estimates_mle[0,11] = std_trans_var_cont_mle 
		estimates_mle[0,12] = std_perm_var_jump_mle
		estimates_mle[0,13] = std_trans_var_jump_mle
		estimates_mle[0,14] = std_perm_var_overnight_mle
		estimates_mle[0,15] = std_trans_var_overnight_mle 
		estimates_mle[0,16] = std_kappa_mle
		estimates_mle[0,17] = std_beta_mle 
		estimates_mle[0,18] = std_si_mle 
		estimates_mle[0,19] = fun
		estimates_mle[0,20] = time_taken
		estimates_mle[0,21] = permno
		estimates_mle[0,22] = series_2['YEAR_MONTH'].iloc[0]
		estimates_mle[0,23] = T_perm_var_cont_mle 
		estimates_mle[0,24] = T_trans_var_cont_mle 
		estimates_mle[0,25] = T_perm_var_jump_mle
		estimates_mle[0,26] = T_trans_var_jump_mle 
		estimates_mle[0,27] = T_perm_var_overnight_mle 
		estimates_mle[0,28] = T_trans_var_overnight_mle 
		estimates_mle[0,29] = T_kappa_mle 
		estimates_mle[0,30] = T_beta_mle 
		estimates_mle[0,31] = T_si_mle 
		estimates_mle[0,32] = std_perm_var_cont_2
		estimates_mle[0,33] = std_trans_var_cont_2
		estimates_mle[0,34] = std_perm_var_jump_2 
		estimates_mle[0,35] = std_trans_var_jump_2 
		estimates_mle[0,36] = std_perm_var_overnight_2 
		estimates_mle[0,37] = std_trans_var_overnight_2 
		estimates_mle[0,38] =T_perm_var_cont_2 
		estimates_mle[0,39] =T_trans_var_cont_2 
		estimates_mle[0,40] =T_perm_var_jump_2 
		estimates_mle[0,41] =T_trans_var_jump_2
		estimates_mle[0,42] =T_perm_var_overnight_2
		estimates_mle[0,43] =T_trans_var_overnight_2 
		estimates_mle[0,44] =T_kappa_2 
		estimates_mle[0,45] =T_beta_2 
		estimates_mle[0,46] =T_si_2 
		estimates_mle[0,47] =len(indicators[indicators==1])
		estimates_mle[0,48] =len(indicators[indicators==0])
		estimates_mle[0,49] =len(indicators[indicators==2])
		estimates_mle[0,50]=np.std(series_2['orderflow_5'].values)
		estimates_mle[0,51] =np.std(series_2['orderflow_5_inno'].values)
		estimates_mle[0,52] = std_kappa_2
		estimates_mle[0,53]=  std_beta_2 
		estimates_mle[0,54]=  std_si_2 
		estimates_mle[0,55]= None
		estimates_mle[0,56]=initial_message
		estimates_mle[0,57]=original_order
		estimates_mle[0,58]= series_2['YEAR_MONTH'].iloc[-1]
		estimates_mle[0,59]= 0



	else:
		estimates_mle[0,0] = 0
		estimates_mle[0,1] = None
		estimates_mle[0,2] = None
		estimates_mle[0,3] = None
		estimates_mle[0,4] = None
		estimates_mle[0,5] = None
		estimates_mle[0,6] = None
		estimates_mle[0,7] = None
		estimates_mle[0,8] = None
		estimates_mle[0,9] = None 
		estimates_mle[0,10] = None 
		estimates_mle[0,11] = None
		estimates_mle[0,12] = None
		estimates_mle[0,13] = None
		estimates_mle[0,14] = None
		estimates_mle[0,15] = None
		estimates_mle[0,16] = None
		estimates_mle[0,17] = None
		estimates_mle[0,18] = None
		estimates_mle[0,19] = None
		estimates_mle[0,20] = None
		estimates_mle[0,21] = permno
		estimates_mle[0,22] = series_2['YEAR_MONTH'].iloc[0]
		estimates_mle[0,23] = None
		estimates_mle[0,24] = None
		estimates_mle[0,25] = None
		estimates_mle[0,26] = None
		estimates_mle[0,27] = None
		estimates_mle[0,28] = None
		estimates_mle[0,29] = None
		estimates_mle[0,30] = None 
		estimates_mle[0,31] = None
		estimates_mle[0,32] = None
		estimates_mle[0,33] = None
		estimates_mle[0,34] = None
		estimates_mle[0,35] = None
		estimates_mle[0,36] = None
		estimates_mle[0,37] = None
		estimates_mle[0,38] =None
		estimates_mle[0,39] =None
		estimates_mle[0,40] =None
		estimates_mle[0,41] =None
		estimates_mle[0,42] =None
		estimates_mle[0,43] =None
		estimates_mle[0,44] =None
		estimates_mle[0,45] =None
		estimates_mle[0,46] =None
		estimates_mle[0,47] =len(indicators[indicators==1])
		estimates_mle[0,48] =len(indicators[indicators==0])
		estimates_mle[0,49] =len(indicators[indicators==2])
		estimates_mle[0,50]=np.std(series_2['orderflow_5'].values)
		estimates_mle[0,51]=np.std(series_2['orderflow_5_inno'].values)
		estimates_mle[0,52] = None
		estimates_mle[0,53]=  None
		estimates_mle[0,54]=  None
		estimates_mle[0,55]= None
		estimates_mle[0,56]=initial_message
		estimates_mle[0,57]=original_order
		estimates_mle[0,58]= series_2['YEAR_MONTH'].iloc[-1]
		estimates_mle[0,59]= 0


	# #===============================================
	#===============================================
	sys.stdout = orig_stdout
	f.close()
	#============================================# #==============================================




	print("label is ", label)
	#========================================================================================= ==
	np.savetxt("/home/khansaad/SP1500/2005/Results/permno_%s.csv"%(label), estimates_mle, delimiter=",")



#LOOP==
global n_obs_count
n_obs_count = len(permno.values)

for i in range(n_obs_count):
	input_permno = permno.iloc[i]
	scanloop(permno=input_permno['PERMNO'],date_q = input_permno['date_q'], original_order= input_permno['original_order'])










