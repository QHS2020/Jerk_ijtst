# -*- coding: utf-8 -*-

"""
This file contain the modules for irregular driving behavior:

    - The longitudinal dim irregularity is represented by the jerk
    - The lateral dim irregularity is represented by the greater wandering.

"""

from RequiredModules import *
from .VehicleModel import IDM
import uuid,builtins
from scipy import integrate
##The following method is 4 plot the heatmap
from scipy.ndimage.filters import gaussian_filter
import matplotlib.cm as cm
def myplot(x, y, sigma= 5, bins=(100, 100)):
    #
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=sigma)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent
#
def hist2distribution(data, bins = 100):
    hs,es = np.histogram(data, bins = bins)
    #return hs,es[1:]
    return hs/sum(hs)/(es[-1] - es[-2]),es[1:]
    
#vehicle parameters, veh_paras['width']
veh_paras = {'lf':1.1, 'lr':1.3, 'lF':2.5, 'lR':2.5, 'width':2.0, 'max_steer':70*np.pi/180.0, 'min_steer':-70*np.pi/180.0, 'max_steer_rate':0.4, 'min_steer_rate':-0.4, }

#idm car following parameters
#   vf unit is m/s; 
idm_paras = {'idm_vf':120.0/3.6, 'idm_T':1.5, 'idm_delta':4.0, 'idm_s0':2.0, 'idm_a':1.0, 'idm_b':3.21, 'a_MAX':3.5, 'veh_len':4}
idm_paras_AV = {'idm_vf':120.0/3.6, 'idm_T':.5, 'idm_delta':4.0, 'idm_s0':2.0, 'idm_a':1.0, 'idm_b':3.21, 'a_MAX':3.5, 'veh_len':4}
#
#page 8 in Ngoduy, D., S. Lee, M. Treiber, M. Keyvan-Ekbatani, and H. L. Vu. 2019. “Langevin Method for a Continuous Stochastic Car-Following Model and Its Stability Conditions.” TRANSPORTATION RESEARCH PART C-EMERGING TECHNOLOGIES 105 (August): 599–610. https://doi.org/10.1016/j.trc.2019.06.005.
#Peng, Guanghan, Hongdi He, and Wei-Zhen Lu. 2016. “A New Car-Following Model with the Consideration of Incorporating Timid and Aggressive Driving Behaviors.” Physica A: Statistical Mechanics and Its Applications 442 (January): 197–202. https://doi.org/10.1016/j.physa.2015.09.009.
OV_paras = {'beta':0.65, 'v0':17.65, 'sc':8.2, 'alpha':1.85, 'sigma0_Ngoduy':0.88, \
                'peng_alpha':0.85, 'peng_V1':6.75, 'peng_V2':7.91, 'peng_C1':0.13, 'peng_C2':1.57, 'peng_lc':5, 'lambda_peng':0.4, }

#two_dim_paras = {'alpha_roadbounds':1.51, 'beta_lane_marks':3.6, 'beta_lane_middle_line':.56, 'sigma_long':.01, 'sigma_lat':.01, 'sigma_long_drift':1.0, 'sigma_lat_drift':1.0, 'gamma_middle_line':5}
two_dim_paras = {'alpha_roadbounds':.1, 'beta_lane_marks': 1.5, 'beta_lane_middle_line':.56, 'sigma_long':.1, 'sigma_lat':.01, 'sigma_long_drift':1.0, 'sigma_lat_drift':1.0, 'gamma_middle_line':1.0, 'theta_ou':.1, 'amplyfier_bound': 1.0, 'amplyfier_lane_mark': 10, 'amplyfier_intra_lanes':30}
#
"""
two_dim_paras_AV = {'alpha_roadbounds': .1, 'beta_lane_marks':3.6, 'beta_lane_middle_line':.56, \
    'sigma_long':.1, 'sigma_lat':.0002, 'sigma_long_drift':1.0, 'sigma_lat_drift':1.0, 'gamma_middle_line':2.0, 'theta_ou':.1, 'amplyfier_lane_mark':10}
"""
two_dim_paras_AV = {'alpha_roadbounds': .1, 'beta_lane_marks':3.6, 'beta_lane_middle_line':.56, \
    'sigma_long':.1, 'sigma_lat':.0002, 'sigma_long_drift':1.0, 'sigma_lat_drift':1.0, \
    'gamma_middle_line':2.0, 'theta_ou':.1, 'amplyfier_bound': 1.0, 'amplyfier_lane_mark': 1.0, \
    'amplyfier_intra_lanes_long': 1e-6, 'amplyfier_intra_lanes_lat':1e-7}
#


class FD_analysis():
    """
    
    """

    @classmethod
    def plot_FDs_given_mprs_results_given_lanes_number_averageoverlane_changecolor(self, FD_mprs_given_lanesnumber, ax = False, figsize = (5,3), alpha = .4, markersize = 10, marker = False):
        """
        Note that this method just change the marker, not the coloer. 
        
        If want to change the color, using :
        
            - self.plot_FDs_given_mprs_results_given_lanes_number_averageoverlane_changecolor()
        ---------------------------------------------------------
        @input: FD_mprs_given_lanesnumber
        
            FD_mprs_given_lanesnumber[mpr][desired_density] = (Q,K,V)
            
            Q is a dict, Q[laneid] = [q1,q2,...qn], eqch qi is a float. 
        
            It is obtained via: 
                #FD_mprs[lanesnumber][mpr][desired_density] = (Q,K,V)
                #Q is a dict, Q[laneid] = [q1,q2,...qn], eqch qi is a float. 
                FD_mprs = pickle.load(open(dataspath + 'RES_FD_mprs.pickle', 'rb'))
            
            and then FD_mprs_given_lanesnumber = FD_mprs[lanesnumber]
            
        

        
        
        """
        #
        if isinstance(marker, bool):
            #color = np.random.uniform(size = (3,))
            marker = np.random.choice(['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_', 'P', 'X', ])
        """
        if isinstance(marker, bool):
            marker = np.random.choice(['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_', 'P', 'X', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,])
        """
        #
        #
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)#.add_subplot(projection='3d')
            
            #ax = plt.figure().add_subplot(projection='3d')
            #ax.set_xlabel('t');ax.set_ylabel('x');ax.set_zlabel('speed'); 
            #ax.grid();
            #ax = host_subplot(111)
        #
        
        
        for mpr in FD_mprs_given_lanesnumber.keys():
            #
            legend_added = True
            #
            color = np.random.uniform(size = (3,))
            #marker = np.random.choice(['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_', 'P', 'X'])
            #
            for desired_density in FD_mprs_given_lanesnumber[mpr].keys():
                #Q[laneid] = [q1,q2,...qn]
                Q,K,V = FD_mprs_given_lanesnumber[mpr][desired_density]
                #
                
                ks = [np.mean(K[laneid]) for laneid in sorted(Q.keys())]
                qs = [np.mean(Q[laneid]) for laneid in sorted(Q.keys())]

                if legend_added:
                #color = np.random.uniform(size = (3,))
                    #
                    ax.plot([np.mean(ks)], [np.mean(qs)], marker = marker, color = color, alpha = alpha, label = 'mpr = ' + str(mpr), markersize = markersize)
                    #
                    legend_added = False
                else:
                    ax.plot([np.mean(ks)], [np.mean(qs)], marker = marker, color = color, alpha = alpha, markersize = markersize)
                    
                    
                    
                    #ax.plot([np.mean(K[laneid])], [np.mean(Q[laneid])],'.', alpha = alpha)
                    
                    """
                    if legend_added:
                    #color = np.random.uniform(size = (3,))
                        #
                        ax.plot([np.mean(K[laneid])], [np.mean(Q[laneid])],'.', color = color, alpha = alpha, label = label)
                        #
                        legend_added = False
                    else:
                        ax.plot([np.mean(K[laneid])], [np.mean(Q[laneid])],'.', color = color, alpha = alpha)
                    """
                
            
            #
        return ax

    @classmethod
    def capacity_from_FD_singlelane(self, FD_density_as_key, typee = 'minimal', only_use_later_percentage = False):
        """
        Calculate the capaity from the FD.
        
        Difference;
            
            self.capacity_from_FD
            self.capacity_from_FD_singlelane
            
        The latter one dont have laneid as key in Q  (FD_density_as_key[density] = (Q,V,K))
        
        
        
        @input: FD_density_as_key
        
            FD_density_as_key[density] = (Q,V,K), Q[lane_id] is a list. 
        
        @intput: typee
        
            typee is either 'mean' or 'minimal'
            
            If is mean, the returned capacity is the average over all moments
            If is minimal, the returned capacity is the minimal flow rate. 
        
        
        @OUTPUT: capacity
            
            a float. 
            
        """
        
        #select
        densities = sorted(FD_density_as_key.keys())
        somedensity = np.random.choice(list(FD_density_as_key.keys()))

        if not isinstance(only_use_later_percentage, bool):
            #FD_multilane[somedensity][0] is Q, a dict, the keys are lane ids. 
            idx = int(len(FD_density_as_key[somedensity][0])*(1-only_use_later_percentage))
        else:
            idx = 0
            
        if typee=='mean':
            #----------------------------------------
            QS = []
            VS = []
            KS = []
            for density in FD_density_as_key.keys():
                #Q[laneid] is a list. 
                Q,V,K = FD_density_as_key[density]
                qs_lanes = [np.mean(Q[idx:])]
                vs_lanes = [np.mean(V[idx:])]
                ks_lanes = [np.mean(K[idx:])]
                
                QS.append(np.mean(qs_lanes))
                VS.append(np.mean(vs_lanes))
                KS.append(np.mean(ks_lanes))
            #
            capacity = max(QS)
            #idx = np.where(np.array(QS)==capacity)[0][0]
        elif typee=='minimal':
            #
            capacity = np.inf
            qs = [np.mean(FD_density_as_key[density][0][idx:]) for density in sorted(FD_density_as_key.keys())]
            vs = [np.mean(FD_density_as_key[density][1][idx:]) for density in sorted(FD_density_as_key.keys())]
            ks = [np.mean(FD_density_as_key[density][2][idx:]) for density in sorted(FD_density_as_key.keys())]
            #
            idx = np.where(np.array(qs)==max(qs))[0][0]
            #print(max(qs))
            #capacity_laneids = min(capacity_laneids, max(qs))
            capacity = min(capacity, min(FD_density_as_key[densities[idx]][0][idx:]))
    
        return capacity
    

    
    @classmethod
    def capacity_from_FD(self, FD_density_as_key, typee = 'minimal', only_use_later_percentage = False):
        """
        Calculate the capaity from the FD.
        
        @input: FD_density_as_key
        
            FD_density_as_key[density] = (Q,V,K), Q[lane_id] is a list. 
        
        @intput: typee
        
            typee is either 'mean' or 'minimal'
            
            If is mean, the returned capacity is the average over all moments
            If is minimal, the returned capacity is the minimal flow rate. 
        
        
        @OUTPUT: capacity
            
            a float. 
            
        """
        
        #select
        densities = sorted(FD_density_as_key.keys())
        somedensity = np.random.choice(list(FD_density_as_key.keys()))
        lane_ids = list(FD_density_as_key[somedensity][0].keys())

        if not isinstance(only_use_later_percentage, bool):
            #FD_multilane[somedensity][0] is Q, a dict, the keys are lane ids. 
            idx = int(len(FD_density_as_key[somedensity][0][lane_ids[0]])*(1-only_use_later_percentage))
        else:
            idx = 0
            
        if typee=='mean':
            #----------------------------------------
            QS = []
            VS = []
            KS = []
            for density in FD_density_as_key.keys():
                #Q[laneid] is a list. 
                Q,V,K = FD_density_as_key[density]
                qs_lanes = [np.mean(Q[laneid][idx:]) for laneid in Q.keys()]
                vs_lanes = [np.mean(V[laneid][idx:]) for laneid in V.keys()]
                ks_lanes = [np.mean(K[laneid][idx:]) for laneid in K.keys()]
                
                QS.append(np.mean(qs_lanes))
                VS.append(np.mean(vs_lanes))
                KS.append(np.mean(ks_lanes))
            #
            capacity = max(QS)
            #idx = np.where(np.array(QS)==capacity)[0][0]
        elif typee=='minimal':
            #
            capacity = np.inf
            #
            for laneid in lane_ids:
                qs = [np.mean(FD_density_as_key[density][0][laneid][idx:]) for density in sorted(FD_density_as_key.keys())]
                vs = [np.mean(FD_density_as_key[density][1][laneid][idx:]) for density in sorted(FD_density_as_key.keys())]
                ks = [np.mean(FD_density_as_key[density][2][laneid][idx:]) for density in sorted(FD_density_as_key.keys())]
                #
                idx = np.where(np.array(qs)==max(qs))[0][0]
                #print(max(qs))
                #capacity_laneids = min(capacity_laneids, max(qs))
                capacity = min(capacity, min(FD_density_as_key[densities[idx]][0][laneid][idx:]))
        
        return capacity
    
    @classmethod
    def CAF_mixedAVs(self, FD_miexedAVs, base_capacity = 1952, only_use_later_percentage = False, exclude_width_threshodl = 2.0, sortt = False):
        """
        @input: FD_miexedAVs
        
            FD_miexedAVs[lanesnumber][mpr][desired_density] = (Q,V,K), Q[laneid] is a list. 
            
            They are obtained via:
            CAF_lanewidth
                reload(CAF)
                #FD_lanewidths[lanesnumber][lanewidth][mpr][desired_density] = (Q,V,K), Q[laneid] is a list. 
                FD_lanewidths = pickle.load(open(dataspath + 'RES_FD_lanewidths_new.pickle', 'rb'))
        @OUTPUT: capacities
            
            Capacities[lanewidth][mpr] = capacity
            
            
            pd.DataFrame(Capacities), rows are mprs and columns are the lane widths. 
        
        
        """
        #Capacities[lanewidth][mpr] = capacity
        Capacities = {}
        #
        #lanewidths and mprs
        lanesNs = sorted(FD_miexedAVs.keys())
        mprs = sorted(FD_miexedAVs[lanesNs[0]].keys())
        densities = sorted(FD_miexedAVs[lanesNs[0]][mprs[0]].keys())
        lane_ids = list(FD_miexedAVs[lanesNs[0]][mprs[0]][densities[0]][0].keys())
        #
        if not isinstance(only_use_later_percentage, bool):
            #FD_lanewidths[somedensity][0] is Q, a dict, the keys are lane ids. 
            idx = int(len(FD_miexedAVs[lanesNs[0]][mprs[0]][densities[0]][0][lane_ids[0]])*(1-only_use_later_percentage))
        else:
            idx = 0
        #
        for lanesN in lanesNs:
            #if lanewidth<exclude_width_threshodl:continue
            Capacities[lanesN] = {}
            for mpr in mprs:
                QS = [];VS = [];KS = []
                for density in FD_miexedAVs[lanesN][mpr].keys():
                    #Q[laneid] is a list. 
                    Q,V,K = FD_miexedAVs[lanesN][mpr][density]
                    #
                    qs_lanes = [np.mean(Q[laneid][idx:]) for laneid in Q.keys()]
                    vs_lanes = [np.mean(V[laneid][idx:]) for laneid in V.keys()]
                    ks_lanes = [np.mean(K[laneid][idx:]) for laneid in K.keys()]
                    #
                    QS.append(np.mean(qs_lanes))
                    VS.append(np.mean(vs_lanes))
                    KS.append(np.mean(ks_lanes))
                    #
                #
                capacity = max(QS)
                #idx = np.where(np.array(QS)==capacity)[0][0]
                #
                Capacities[lanesN][mpr] = capacity
            
        #Capacities.loc[mpr, lanewidth]
        Capacities = pd.DataFrame(Capacities)
        Capacities_returned = copy.deepcopy(Capacities)
        
        if sortt:
            for mpr in Capacities_returned.index:
                Capacities_returned.loc[mpr, :] = list(sorted(Capacities.loc[mpr, :].values))
            
        return Capacities_returned
    
    
    
    @classmethod
    def CAF_lanewidth(self, FD_lanewidths, base_capacity = 1952, only_use_later_percentage = False, exclude_width_threshodl = 2.0, sortt = False):
        """
        @input: FD_lanewidths
        
            FD_lanewidths[lanewidth][mpr][desired_density] = (Q,V,K), Q[laneid] is a list. 
            
            They are obtained via:
            CAF_lanewidth
                reload(CAF)
                #FD_lanewidths[lanesnumber][lanewidth][mpr][desired_density] = (Q,V,K), Q[laneid] is a list. 
                FD_lanewidths = pickle.load(open(dataspath + 'RES_FD_lanewidths_new.pickle', 'rb'))
        @OUTPUT: capacities
            
            Capacities[lanewidth][mpr] = capacity
            
            
            pd.DataFrame(Capacities), rows are mprs and columns are the lane widths. 
        
        
        """
        #Capacities[lanewidth][mpr] = capacity
        Capacities = {}
        #
        #lanewidths and mprs
        lanewidths = sorted(FD_lanewidths.keys())
        mprs = sorted(FD_lanewidths[lanewidths[0]].keys())
        densities = sorted(FD_lanewidths[lanewidths[0]][mprs[0]].keys())
        lane_ids = list(FD_lanewidths[lanewidths[0]][mprs[0]][densities[0]][0].keys())
        #
        if not isinstance(only_use_later_percentage, bool):
            #FD_lanewidths[somedensity][0] is Q, a dict, the keys are lane ids. 
            idx = int(len(FD_lanewidths[lanewidths[0]][mprs[0]][densities[0]][0][lane_ids[0]])*(1-only_use_later_percentage))
        else:
            idx = 0
        #
        for lanewidth in lanewidths:
            if lanewidth<exclude_width_threshodl:continue
            Capacities[lanewidth] = {}
            for mpr in mprs:
                QS = [];VS = [];KS = []
                for density in FD_lanewidths[lanewidth][mpr].keys():
                    #Q[laneid] is a list. 
                    Q,V,K = FD_lanewidths[lanewidth][mpr][density]
                    #
                    qs_lanes = [np.mean(Q[laneid][idx:]) for laneid in Q.keys()]
                    vs_lanes = [np.mean(V[laneid][idx:]) for laneid in V.keys()]
                    ks_lanes = [np.mean(K[laneid][idx:]) for laneid in K.keys()]
                    #
                    QS.append(np.mean(qs_lanes))
                    VS.append(np.mean(vs_lanes))
                    KS.append(np.mean(ks_lanes))
                    #
                #
                capacity = max(QS)
                #idx = np.where(np.array(QS)==capacity)[0][0]
                #
                Capacities[lanewidth][mpr] = capacity
            
        #Capacities.loc[mpr, lanewidth]
        Capacities = pd.DataFrame(Capacities)
        Capacities_returned = copy.deepcopy(Capacities)
        
        if sortt:
            for mpr in Capacities_returned.index:
                Capacities_returned.loc[mpr, :] = list(sorted(Capacities.loc[mpr, :].values))
            
        return Capacities_returned
        
    
    @classmethod
    def CAF_averageoverlanes(self, FD_multilane, base_capacity = 1952, only_use_later_percentage = False):
        """
        calculate the CAF. The 
        
        @input: FD_multilane
        
            FD_multilane[desireddensity] = (Q,V,K), Q[lane_id] is a list. 
        
        @input: base_capacity
        
            unit is veh/h. 
        
        ------------------------------------------------
        @Steps:
            
            - For each density, calcuate
        
        
        """
        #select
        somedensity = np.random.choice(list(FD_multilane.keys()))
        lane_ids = list(FD_multilane[somedensity][0].keys())

        if not isinstance(only_use_later_percentage, bool):
            #FD_multilane[somedensity][0] is Q, a dict, the keys are lane ids. 
            idx = int(len(FD_multilane[somedensity][0][lane_ids[0]])*(1-only_use_later_percentage))
        else:
            idx = 0
            
            
        #----------------------------------------
        QS = []
        VS = []
        KS = []
        for density in FD_multilane.keys():
            #Q[laneid] is a list. 
            Q,V,K = FD_multilane[density]
            qs_lanes = [np.mean(Q[laneid][idx:]) for laneid in Q.keys()]
            vs_lanes = [np.mean(V[laneid][idx:]) for laneid in V.keys()]
            ks_lanes = [np.mean(K[laneid][idx:]) for laneid in K.keys()]
            
            QS.append(np.mean(qs_lanes))
            VS.append(np.mean(vs_lanes))
            KS.append(np.mean(ks_lanes))
            
            
            pass
        #
        capacity = max(QS)
        idx = np.where(np.array(QS)==capacity)[0][0]
        
        
        return 1.0*capacity/base_capacity,KS[idx]
        

    @classmethod
    def CAF_by_minimal_of_lane_cap(self, FD_multilane, base_capacity = 1952, only_use_later_percentage = False):
        """
        calculate the CAF. The 
        
        @input: FD_multilane
        
            FD_multilane[desireddensity] = (Q,V,K), Q[lane_id] is a list. 
        
        @input: base_capacity
        
            unit is veh/h. 
        
        ------------------------------------------------
        @Steps:
            
            - For each density, calcuate
        
        
        """
        #select
        densities = sorted(FD_multilane.keys())
        somedensity = np.random.choice(list(FD_multilane.keys()))
        #
        lane_ids = list(FD_multilane[somedensity][0].keys())

        if not isinstance(only_use_later_percentage, bool):
            #FD_multilane[somedensity][0] is Q, a dict, the keys are lane ids. 
            idx = int(len(FD_multilane[somedensity][0][lane_ids[0]])*(1-only_use_later_percentage))
        else:
            idx = 0
            
            
        #----------------------------------------

        capacity_laneids = np.inf
        #
        for laneid in lane_ids:
            qs = [np.mean(FD_multilane[density][0][laneid][idx:]) for density in sorted(FD_multilane.keys())]
            vs = [np.mean(FD_multilane[density][1][laneid][idx:]) for density in sorted(FD_multilane.keys())]
            ks = [np.mean(FD_multilane[density][2][laneid][idx:]) for density in sorted(FD_multilane.keys())]
            #
            idx = np.where(np.array(qs)==max(qs))[0][0]
            #print(max(qs))
            #capacity_laneids = min(capacity_laneids, max(qs))
            capacity_laneids = min(capacity_laneids, min(FD_multilane[densities[idx]][0][laneid][idx:]))
            
            
        #
        return 1.0*capacity_laneids/base_capacity


    @classmethod
    def plot_FD_multilane_qsvs_distribution_ARCHIEVE(self, FD_multilane, ax = False, figsize = (5,3), alpha = .4, markersize = 10, color = False, marker = 's', only_use_later_percentage = False):
        """
        
        
        @input: FD_multilane

            FD_multilane[desireddensity] = (Q,V,K), Q[lane_id] is a list. 

                reload(CAF)
                #FD_multilanes[lanesnumber][desireddensity] = (Q,V,K), Q[lane_id] is a list. 
                FD_multilanes = pickle.load(open(dataspath + 'RES_FD_multilane.pickle', 'rb'))
            
                FD_multilane = FD_multilanes[2]
        
        @input: only_use_later_percentage
        
            a float within (0, 1)
            
            
        
        
        """
        #select
        somedensity = np.random.choice(list(FD_multilane.keys()))
        lane_ids = list(FD_multilane[somedensity][0].keys())
        colors = [np.random.uniform(size = (3,)) for i in lane_ids]
        #if isinstance(color,bool):
        #    color = np.random.uniform(size = (3,))
        #
        markers = np.random.choice([',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_', 'P', 'X', ], len(lane_ids))
        
        #
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)#.add_subplot(projection='3d')

        if not isinstance(only_use_later_percentage, bool):
            #FD_multilane[somedensity][0] is Q, a dict, the keys are lane ids. 
            idx = int(len(FD_multilane[somedensity][0][lane_ids[0]])*(1-only_use_later_percentage))
        else:
            idx = 0
        #
        
        #
        for laneid,color,marker in zip(lane_ids, colors,markers):
            #
            legend_added = True
            for density in FD_multilane.keys():
                qs = FD_multilane[density][0][laneid][idx:]
                vs = np.array(FD_multilane[density][1][laneid][idx:])
                ks = FD_multilane[density][2][laneid][idx:]
                #
                if legend_added:
                #color = np.random.uniform(size = (3,))
                    #
                    #ax.plot(qs, 2*vs/1.609344, marker = marker, color = color, alpha = alpha, markersize = markersize, label = str(laneid))
                    ax.plot(qs, 2*vs, marker = marker, color = color, alpha = alpha, markersize = markersize, label = str(laneid))
                    #
                    legend_added = False
                else:
                    #ax.plot(qs, 2*vs/1.609344, marker = marker, color = color, alpha = alpha, markersize = markersize,)
                    ax.plot(qs, 2*vs, marker = marker, color = color, alpha = alpha, markersize = markersize,)
                 
            """
            qs = [np.mean(FD_multilane[density][0][laneid][idx:]) for density in FD_multilane.keys()]
            vs = np.array([np.mean(FD_multilane[density][1][laneid][idx:]) for density in FD_multilane.keys()])
            ks = [np.mean(FD_multilane[density][2][laneid][idx:]) for density in FD_multilane.keys()]
            #
            ax.plot(qs, 2*vs, marker = marker, color = color, alpha = alpha, markersize = markersize, label = str(laneid))
            """
            
            """
            if legend_added:
            #color = np.random.uniform(size = (3,))
                #
                ax.plot([np.mean(ks)], [np.mean(qs)], marker = marker, color = color, alpha = alpha, label = 'mpr = ' + str(mpr), markersize = markersize)
                #
                legend_added = False
            else:
                ax.plot([np.mean(ks)], [np.mean(qs)], marker = marker, color = color, alpha = alpha, markersize = markersize)
            
            """
        ax.set_ylabel('speed ( km/h )');ax.set_xlabel('Q (veh/h) ')
        #
        return ax


    @classmethod
    def plot_FD_multilane_qsvs_ARCHIEVE(self, FD_multilane, ax = False, figsize = (5,3), alpha = .4, markersize = 10, color = False, marker = 's', only_use_later_percentage = False):
        """
        
        
        @input: FD_multilane

            FD_multilane[desireddensity] = (Q,V,K), Q[lane_id] is a list. 

                reload(CAF)
                #FD_multilanes[lanesnumber][desireddensity] = (Q,V,K), Q[lane_id] is a list. 
                FD_multilanes = pickle.load(open(dataspath + 'RES_FD_multilane.pickle', 'rb'))
            
                FD_multilane = FD_multilanes[2]
        
        @input: only_use_later_percentage
        
            a float within (0, 1)
            
            
        
        
        """
        #select
        somedensity = np.random.choice(list(FD_multilane.keys()))
        lane_ids = list(FD_multilane[somedensity][0].keys())
        colors = [np.random.uniform(size = (3,)) for i in lane_ids]
        #if isinstance(color,bool):
        #    color = np.random.uniform(size = (3,))
        #
        markers = np.random.choice([',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_', 'P', 'X', ], len(lane_ids))
        
        
        #
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)#.add_subplot(projection='3d')

        if not isinstance(only_use_later_percentage, bool):
            #FD_multilane[somedensity][0] is Q, a dict, the keys are lane ids. 
            idx = int(len(FD_multilane[somedensity][0][lane_ids[0]])*(1-only_use_later_percentage))
        else:
            idx = 0
        #
        
        #
        for laneid,color,marker in zip(lane_ids, colors,markers):
            

            qs = [np.mean(FD_multilane[density][0][laneid][idx:]) for density in FD_multilane.keys()]
            vs = np.array([np.mean(FD_multilane[density][1][laneid][idx:]) for density in FD_multilane.keys()])
            ks = [np.mean(FD_multilane[density][2][laneid][idx:]) for density in FD_multilane.keys()]
            #
            ax.plot(qs, 2*vs, marker = marker, color = color, alpha = alpha, markersize = markersize, label = str(laneid))
            #ax.plot(qs, 2*vs/1.609344, marker = marker, color = color, alpha = alpha, markersize = markersize, label = str(laneid))
            
            """
            if legend_added:
            #color = np.random.uniform(size = (3,))
                #
                ax.plot([np.mean(ks)], [np.mean(qs)], marker = marker, color = color, alpha = alpha, label = 'mpr = ' + str(mpr), markersize = markersize)
                #
                legend_added = False
            else:
                ax.plot([np.mean(ks)], [np.mean(qs)], marker = marker, color = color, alpha = alpha, markersize = markersize)
            
            """
        ax.set_ylabel('speed ( km/h )');ax.set_xlabel('Q (veh/h) ')
        #
        return ax


    @classmethod
    def plot_FD_multilane_qsvs(self, FD_multilane, ax = False, figsize = (5,3), alpha = .4, markersize = 10, color = False, marker = 's', only_use_later_percentage = False):
        """
        
        
        @input: FD_multilane

            FD_multilane[desireddensity] = (Q,V,K), Q[lane_id] is a list. 

                reload(CAF)
                #FD_multilanes[lanesnumber][desireddensity] = (Q,V,K), Q[lane_id] is a list. 
                FD_multilanes = pickle.load(open(dataspath + 'RES_FD_multilane.pickle', 'rb'))
            
                FD_multilane = FD_multilanes[2]
        
        @input: only_use_later_percentage
        
            a float within (0, 1)
            
            
        
        
        """
        #select
        somedensity = np.random.choice(list(FD_multilane.keys()))
        lane_ids = list(FD_multilane[somedensity][0].keys())
        colors = [np.random.uniform(size = (3,)) for i in lane_ids]
        #if isinstance(color,bool):
        #    color = np.random.uniform(size = (3,))
        #
        markers = np.random.choice([',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_', 'P', 'X', ], len(lane_ids))
        
        
        #
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)#.add_subplot(projection='3d')

        if not isinstance(only_use_later_percentage, bool):
            #FD_multilane[somedensity][0] is Q, a dict, the keys are lane ids. 
            idx = int(len(FD_multilane[somedensity][0][lane_ids[0]])*(1-only_use_later_percentage))
        else:
            idx = 0
        #
        
        #
        for laneid,color,marker in zip(lane_ids, colors,markers):
            

            qs = [np.mean(FD_multilane[density][0][laneid][idx:]) for density in FD_multilane.keys()]
            vs = np.array([np.mean(FD_multilane[density][1][laneid][idx:]) for density in FD_multilane.keys()])
            ks = [np.mean(FD_multilane[density][2][laneid][idx:]) for density in FD_multilane.keys()]
            #
            ax.plot(qs, vs, marker = marker, color = color, alpha = alpha, markersize = markersize, label = str(laneid))
            
            """
            if legend_added:
            #color = np.random.uniform(size = (3,))
                #
                ax.plot([np.mean(ks)], [np.mean(qs)], marker = marker, color = color, alpha = alpha, label = 'mpr = ' + str(mpr), markersize = markersize)
                #
                legend_added = False
            else:
                ax.plot([np.mean(ks)], [np.mean(qs)], marker = marker, color = color, alpha = alpha, markersize = markersize)
            
            """
        ax.set_ylabel('speed ( km/h )');ax.set_xlabel('Q (veh/h) ')
        #
        return ax


    @classmethod
    def plot_FD_multilane_distribution(self, FD_multilane, ax = False, figsize = (5,3), alpha = .4, markersize = 10, color = False, marker = 's', only_use_later_percentage = False):
        """
        
        ---------------------------------------
        @input: FD_multilane

            FD_multilane[desireddensity] = (Q,V,K), Q[lane_id] is a list. 

                reload(CAF)
                #FD_multilanes[lanesnumber][desireddensity] = (Q,V,K), Q[lane_id] is a list. 
                FD_multilanes = pickle.load(open(dataspath + 'RES_FD_multilane.pickle', 'rb'))
            
                FD_multilane = FD_multilanes[2]
        
        @input: only_use_later_percentage
        
            a float within (0, 1)
            
            
        
        
        """
        #select
        somedensity = np.random.choice(list(FD_multilane.keys()))
        lane_ids = list(FD_multilane[somedensity][0].keys())
        #colors = [np.random.uniform(size = (3,)) for i in lane_ids]
        colors = ['C'+str(idx) for idx in range(len(lane_ids))]
        #if isinstance(color,bool):
        #    color = np.random.uniform(size = (3,))
        #
        markers = np.random.choice([',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_', 'P', 'X', ], len(lane_ids))
        
        #
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)#.add_subplot(projection='3d')

        if not isinstance(only_use_later_percentage, bool):
            #FD_multilane[somedensity][0] is Q, a dict, the keys are lane ids. 
            idx = int(len(FD_multilane[somedensity][0][lane_ids[0]])*(1-only_use_later_percentage))
        else:
            idx = 0
        #
        
        #
        for laneid,color,marker in zip(lane_ids, colors,markers):
            label_added = True
            for density in FD_multilane.keys():
                qs = FD_multilane[density][0][laneid][idx:]
                vs = FD_multilane[density][1][laneid][idx:]
                ks = FD_multilane[density][2][laneid][idx:]
                #
                if label_added:
                    ax.plot(ks, qs, marker = marker, color = color, alpha = alpha, markersize = markersize, label = str(laneid))
                    label_added = False
                else:
                    
                    ax.plot(ks, qs, marker = marker, color = color, alpha = alpha, markersize = markersize)
                
                
                pass
            
            
            """
            qs = [np.mean(FD_multilane[density][0][laneid][idx:]) for density in FD_multilane.keys()]
            vs = [np.mean(FD_multilane[density][1][laneid][idx:]) for density in FD_multilane.keys()]
            ks = [np.mean(FD_multilane[density][2][laneid][idx:]) for density in FD_multilane.keys()]
            #
            ax.plot(ks, qs, marker = marker, color = color, alpha = alpha, markersize = markersize, label = str(laneid))
            """
            
            """
            if legend_added:
            #color = np.random.uniform(size = (3,))
                #
                ax.plot([np.mean(ks)], [np.mean(qs)], marker = marker, color = color, alpha = alpha, label = 'mpr = ' + str(mpr), markersize = markersize)
                #
                legend_added = False
            else:
                ax.plot([np.mean(ks)], [np.mean(qs)], marker = marker, color = color, alpha = alpha, markersize = markersize)
            
            """
        return ax


    @classmethod
    def plot_Q_timeseries_laneaverage(self, FD_multilane, ax = False, figsize = (5,3), alpha = .4, markersize = 10, color = False, marker = 's', only_use_later_percentage = False, exclude_lane = False, label = ' '):
        """
        
        
        @input: FD_multilane

            FD_multilane[desireddensity] = (Q,V,K), Q[lane_id] is a list. 

                reload(CAF)
                #FD_multilanes[lanesnumber][desireddensity] = (Q,V,K), Q[lane_id] is a list. 
                FD_multilanes = pickle.load(open(dataspath + 'RES_FD_multilane.pickle', 'rb'))
            
                FD_multilane = FD_multilanes[2]
        
        @input: only_use_later_percentage
        
            a float within (0, 1)
            
            
        
        
        """
        
        #
        somedensity = np.random.choice(list(FD_multilane.keys()))
        lane_ids = list(FD_multilane[somedensity][0].keys())
        #colors = [np.random.uniform(size = (3,)) for i in lane_ids]
        colors = ['C'+str(idx) for idx in range(len(lane_ids))]
        #if isinstance(color,bool):
        #    color = np.random.uniform(size = (3,))
        #
        markers = np.random.choice([',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_', 'P', 'X', ], len(lane_ids))
        
        #
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)#.add_subplot(projection='3d')

        if not isinstance(only_use_later_percentage, bool):
            #FD_multilane[somedensity][0] is Q, a dict, the keys are lane ids. 
            idx = int(len(FD_multilane[somedensity][0][lane_ids[0]])*(1-only_use_later_percentage))
        else:
            idx = 0
        #
        VS = []
        KS = []
        for density in sorted(FD_multilane.keys()):
            #
            vs = 0*np.array(FD_multilane[density][1][lane_ids[0]][idx:])
            ks = 0*np.array(FD_multilane[density][2][lane_ids[0]][idx:])
            for laneid,_,marker in zip(lane_ids, colors,markers):
                if laneid==exclude_lane:continue
                vs = vs + np.array(FD_multilane[density][1][laneid][idx:])
                ks = ks + np.array(FD_multilane[density][2][laneid][idx:])
            #ks length is [idx:]
            qs  = ks*vs/len(lane_ids)//len(lane_ids)
            #
            #ax.plot(range(len(qs)), qs, marker = marker, color = color, alpha = alpha, markersize = markersize, label = str(laneid))
            #ax.plot([density]*len(qs), qs, marker = marker, color = color, alpha = alpha, markersize = markersize, label = str(laneid))
            ax.plot([density]*len(qs), qs, marker = marker, color = color, alpha = alpha, markersize = markersize, )
            #
            VS.append(np.mean(vs)/len(lane_ids))
            KS.append(np.mean(ks)/len(lane_ids))
            #
        ax.plot(sorted(FD_multilane.keys()), np.array(VS)*np.array(KS),label = label, linewidth = 4)
        
        return ax
        
        
        
        
        #
        for laneid,color,marker in zip(lane_ids, colors,markers):
            """
            qs = [np.mean(FD_multilane[density][0][laneid][idx:]) for density in FD_multilane.keys()]
            vs = [np.mean(FD_multilane[density][1][laneid][idx:]) for density in FD_multilane.keys()]
            ks = [np.mean(FD_multilane[density][2][laneid][idx:]) for density in FD_multilane.keys()]
            """
            for density in FD_multilane.keys():
                qs  =FD_multilane[density][0][laneid][idx:]
                #
                ax.plot(range(len(qs)), qs, marker = marker, color = color, alpha = alpha, markersize = markersize, label = str(laneid))
            
            """
            if legend_added:
            #color = np.random.uniform(size = (3,))
                #
                ax.plot([np.mean(ks)], [np.mean(qs)], marker = marker, color = color, alpha = alpha, label = 'mpr = ' + str(mpr), markersize = markersize)
                #
                legend_added = False
            else:
                ax.plot([np.mean(ks)], [np.mean(qs)], marker = marker, color = color, alpha = alpha, markersize = markersize)
            
            """
        return ax
        
        pass
    


    @classmethod
    def plot_Q_timeseries(self, FD_multilane, ax = False, figsize = (5,3), alpha = .4, markersize = 10, color = False, marker = 's', only_use_later_percentage = False):
        """
        
        
        @input: FD_multilane

            FD_multilane[desireddensity] = (Q,V,K), Q[lane_id] is a list. 

                reload(CAF)
                #FD_multilanes[lanesnumber][desireddensity] = (Q,V,K), Q[lane_id] is a list. 
                FD_multilanes = pickle.load(open(dataspath + 'RES_FD_multilane.pickle', 'rb'))
            
                FD_multilane = FD_multilanes[2]
        
        @input: only_use_later_percentage
        
            a float within (0, 1)
            
            
        
        
        """
        
        #
        somedensity = np.random.choice(list(FD_multilane.keys()))
        lane_ids = list(FD_multilane[somedensity][0].keys())
        #colors = [np.random.uniform(size = (3,)) for i in lane_ids]
        colors = ['C'+str(idx) for idx in range(len(lane_ids))]
        #if isinstance(color,bool):
        #    color = np.random.uniform(size = (3,))
        #
        markers = np.random.choice([',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_', 'P', 'X', ], len(lane_ids))
        
        #
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)#.add_subplot(projection='3d')

        if not isinstance(only_use_later_percentage, bool):
            #FD_multilane[somedensity][0] is Q, a dict, the keys are lane ids. 
            idx = int(len(FD_multilane[somedensity][0][lane_ids[0]])*(1-only_use_later_percentage))
        else:
            idx = 0
        #
        
        #
        for laneid,color,marker in zip(lane_ids, colors,markers):
            """
            qs = [np.mean(FD_multilane[density][0][laneid][idx:]) for density in FD_multilane.keys()]
            vs = [np.mean(FD_multilane[density][1][laneid][idx:]) for density in FD_multilane.keys()]
            ks = [np.mean(FD_multilane[density][2][laneid][idx:]) for density in FD_multilane.keys()]
            """
            for density in FD_multilane.keys():
                qs  =FD_multilane[density][0][laneid][idx:]
                #
                ax.plot(range(len(qs)), qs, marker = marker, color = color, alpha = alpha, markersize = markersize, label = str(laneid))
            
            """
            if legend_added:
            #color = np.random.uniform(size = (3,))
                #
                ax.plot([np.mean(ks)], [np.mean(qs)], marker = marker, color = color, alpha = alpha, label = 'mpr = ' + str(mpr), markersize = markersize)
                #
                legend_added = False
            else:
                ax.plot([np.mean(ks)], [np.mean(qs)], marker = marker, color = color, alpha = alpha, markersize = markersize)
            
            """
        return ax
        
        pass
    

    @classmethod
    def plot_FD_multilane_lanes_total(self, FD_multilane, ax = False, figsize = (5,3), alpha = .4, markersize = 10, color = False, marker = 's', only_use_later_percentage = False , label = ' ', exclude_lane = False):
        """
        @input: exclude_lane
        
            certain lane is dedicated lane, whose density may be fixed. 
            
            Thus to calculate the FD, this lane is excluded. 
            
        
        @input: FD_multilane

            FD_multilane[desireddensity] = (Q,V,K), Q[lane_id] is a list. 

                reload(CAF)
                #FD_multilanes[lanesnumber][desireddensity] = (Q,V,K), Q[lane_id] is a list. 
                FD_multilanes = pickle.load(open(dataspath + 'RES_FD_multilane.pickle', 'rb'))
            
                FD_multilane = FD_multilanes[2]
        
        @input: only_use_later_percentage
        
            a float within (0, 1)
            
            
        
        
        """
        #select
        somedensity = np.random.choice(list(FD_multilane.keys()))
        lane_ids = list(FD_multilane[somedensity][0].keys())
        #colors = [np.random.uniform(size = (3,)) for i in lane_ids]
        colors = ['C'+str(idx) for idx in range(len(lane_ids))]
        #if isinstance(color,bool):
        #    color = np.random.uniform(size = (3,))
        #
        markers = np.random.choice([',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_', 'P', 'X', ], len(lane_ids))
        
        #
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)#.add_subplot(projection='3d')
            

        if not isinstance(only_use_later_percentage, bool):
            #FD_multilane[somedensity][0] is Q, a dict, the keys are lane ids. 
            idx = int(len(FD_multilane[somedensity][0][lane_ids[0]])*(1-only_use_later_percentage))
        else:
            idx = 0
        #
        QS = []
        VS = []
        KS = []
        #
        densities = []
        for density in FD_multilane.keys():
            #
            vs = []
            ks = []
            qs = []
            for laneid,color,marker in zip(lane_ids, colors,markers):
                if laneid==exclude_lane:continue
                #print(np.mean(FD_multilane[density][1][laneid][idx:]))
                vs.append(np.mean(FD_multilane[density][1][laneid][idx:]))
                ks.append(np.mean(FD_multilane[density][2][laneid][idx:]))
                #
                #qs.append(np.mean(FD_multilane[density][0][laneid][idx:]))
                qs.append(vs[-1]*ks[-1])
            """
            qs = [np.mean(FD_multilane[density][0][laneid][idx:]) for laneid,color,marker in zip(lane_ids, colors,markers)]
            vs = [np.mean(FD_multilane[density][1][laneid][idx:]) for laneid,color,marker in zip(lane_ids, colors,markers)]
            ks = [np.mean(FD_multilane[density][2][laneid][idx:]) for laneid,color,marker in zip(lane_ids, colors,markers)]
            """
            
            #QS.append(np.mean(qs))
            VS.append(np.mean(vs))
            KS.append(np.mean(ks))
            QS.append(sum(qs))
            densities.append(density)
        #
        #ax.plot(QS,  VS, alpha = alpha, markersize = markersize, label = label)
        ax.plot(densities, np.array(QS)/4.0,  alpha = alpha, markersize = markersize, label = label)
        ax.set_xlabel('K (veh/km)');ax.set_ylabel('Q (veh/h)');
        
        return ax



    @classmethod
    def plot_FD_multilane_averageoverlane(self, FD_multilane, ax = False, figsize = (5,3), alpha = .4, markersize = 10, color = False, marker = 's', only_use_later_percentage = False , label = ' ', exclude_lane = False):
        """
        @input: exclude_lane
        
            certain lane is dedicated lane, whose density may be fixed. 
            
            Thus to calculate the FD, this lane is excluded. 
            
        
        @input: FD_multilane

            FD_multilane[desireddensity] = (Q,V,K), Q[lane_id] is a list. 

                reload(CAF)
                #FD_multilanes[lanesnumber][desireddensity] = (Q,V,K), Q[lane_id] is a list. 
                FD_multilanes = pickle.load(open(dataspath + 'RES_FD_multilane.pickle', 'rb'))
            
                FD_multilane = FD_multilanes[2]
        
        @input: only_use_later_percentage
        
            a float within (0, 1)
            
            
        
        
        """
        #select
        somedensity = np.random.choice(list(FD_multilane.keys()))
        lane_ids = list(FD_multilane[somedensity][0].keys())
        #colors = [np.random.uniform(size = (3,)) for i in lane_ids]
        colors = ['C'+str(idx) for idx in range(len(lane_ids))]
        #if isinstance(color,bool):
        #    color = np.random.uniform(size = (3,))
        #
        markers = np.random.choice([',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_', 'P', 'X', ], len(lane_ids))
        
        #
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)#.add_subplot(projection='3d')
            

        if not isinstance(only_use_later_percentage, bool):
            #FD_multilane[somedensity][0] is Q, a dict, the keys are lane ids. 
            idx = int(len(FD_multilane[somedensity][0][lane_ids[0]])*(1-only_use_later_percentage))
        else:
            idx = 0
        #
        QS = []
        VS = []
        KS = []
        densities = []
        for density in FD_multilane.keys():
            #
            vs = []
            ks = []
            for laneid,color,marker in zip(lane_ids, colors,markers):
                if laneid==exclude_lane:continue
                vs.extend(FD_multilane[density][1][laneid][idx:])
                ks.extend(FD_multilane[density][2][laneid][idx:])
            """
            qs = [np.mean(FD_multilane[density][0][laneid][idx:]) for laneid,color,marker in zip(lane_ids, colors,markers)]
            vs = [np.mean(FD_multilane[density][1][laneid][idx:]) for laneid,color,marker in zip(lane_ids, colors,markers)]
            ks = [np.mean(FD_multilane[density][2][laneid][idx:]) for laneid,color,marker in zip(lane_ids, colors,markers)]
            """
            
            #QS.append(np.mean(qs))
            VS.append(np.mean(vs))
            KS.append(np.mean(ks))
            QS.append(np.mean(ks)*np.mean(vs))
            densities.append(density)
        #
        #ax.plot(QS,  VS, alpha = alpha, markersize = markersize, label = label)
        ax.plot(densities, QS,  alpha = alpha, markersize = markersize, label = label)
        ax.set_xlabel('K (veh/km)');ax.set_ylabel('Q (veh/h)');
        
        return ax
        
        
        for laneid,color,marker in zip(lane_ids, colors,markers):
            
            qs = [np.mean(FD_multilane[density][0][laneid][idx:]) for density in FD_multilane.keys()]
            vs = [np.mean(FD_multilane[density][1][laneid][idx:]) for density in FD_multilane.keys()]
            ks = [np.mean(FD_multilane[density][2][laneid][idx:]) for density in FD_multilane.keys()]
            #
            ax.plot(ks, qs, marker = marker, color = color, alpha = alpha, markersize = markersize, label = str(laneid))
            
            """
            if legend_added:
            #color = np.random.uniform(size = (3,))
                #
                ax.plot([np.mean(ks)], [np.mean(qs)], marker = marker, color = color, alpha = alpha, label = 'mpr = ' + str(mpr), markersize = markersize)
                #
                legend_added = False
            else:
                ax.plot([np.mean(ks)], [np.mean(qs)], marker = marker, color = color, alpha = alpha, markersize = markersize)
            
            """
        return ax



    @classmethod
    def plot_FD_multilane(self, FD_multilane, ax = False, figsize = (5,3), alpha = .4, markersize = 10, color = False, marker = 's', only_use_later_percentage = False):
        """
        
        
        @input: FD_multilane

            FD_multilane[desireddensity] = (Q,V,K), Q[lane_id] is a list. 

                reload(CAF)
                #FD_multilanes[lanesnumber][desireddensity] = (Q,V,K), Q[lane_id] is a list. 
                FD_multilanes = pickle.load(open(dataspath + 'RES_FD_multilane.pickle', 'rb'))
            
                FD_multilane = FD_multilanes[2]
        
        @input: only_use_later_percentage
        
            a float within (0, 1)
            
            
        
        
        """
        #select
        somedensity = np.random.choice(list(FD_multilane.keys()))
        lane_ids = list(FD_multilane[somedensity][0].keys())
        #colors = [np.random.uniform(size = (3,)) for i in lane_ids]
        colors = ['C'+str(idx) for idx in range(len(lane_ids))]
        #if isinstance(color,bool):
        #    color = np.random.uniform(size = (3,))
        #
        markers = np.random.choice([',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_', 'P', 'X', ], len(lane_ids))
        
        #
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)#.add_subplot(projection='3d')

        if not isinstance(only_use_later_percentage, bool):
            #FD_multilaneplot_FD_multilane_averageoverlane[somedensity][0] is Q, a dict, the keys are lane ids. 
            idx = int(len(FD_multilane[somedensity][0][lane_ids[0]])*(1-only_use_later_percentage))
        else:
            idx = 0
        #
        
        #
        for laneid,color,marker in zip(lane_ids, colors,markers):
            
            qs = [np.mean(FD_multilane[density][0][laneid][idx:]) for density in FD_multilane.keys()]
            vs = [np.mean(FD_multilane[density][1][laneid][idx:]) for density in FD_multilane.keys()]
            ks = [np.mean(FD_multilane[density][2][laneid][idx:]) for density in FD_multilane.keys()]
            #
            ax.plot(ks, qs, marker = marker, color = color, alpha = alpha, markersize = markersize, label = str(laneid))
            
            """
            if legend_added:
            #color = np.random.uniform(size = (3,))
                #
                ax.plot([np.mean(ks)], [np.mean(qs)], marker = marker, color = color, alpha = alpha, label = 'mpr = ' + str(mpr), markersize = markersize)
                #
                legend_added = False
            else:
                ax.plot([np.mean(ks)], [np.mean(qs)], marker = marker, color = color, alpha = alpha, markersize = markersize)
            
            """
        return ax


    @classmethod
    def plot_FD_singlelane_distribution(self, FD_singlelane, ax = False, figsize = (5,3), alpha = .4, markersize = 10, color = False, marker = 's', only_use_later_percentage = False):
        """
        
        
        @input: FD_singlelane
            
            FD_singlelane[]
        
            
            reload(CAF)
            #FD_singlelane[desireddensity] = (Q,V,K), Q V K are all list. 
            FD_singlelane = pickle.load(open(dataspath + 'RES_FD_singlelane.pickle', 'rb'))
        
        
        
        @input: only_use_later_percentage
        
            a float within (0, 1)
            
            
        
        
        """
        #
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)#.add_subplot(projection='3d')
        #
        if isinstance(color, bool):
            color = np.random.uniform(size = (3,))
        for desired_density in FD_singlelane.keys():
            #qs = [q1,q2,...qn]
            if isinstance(only_use_later_percentage, bool):
                qs,vs,ks = FD_singlelane[desired_density]
            else:
                qs0,vs0,ks0 = FD_singlelane[desired_density]
                #
                idx = int(len(qs0)*(1-only_use_later_percentage))
                #
                qs = qs0[idx:]
                vs = vs0[idx:]
                ks = ks0[idx:]
                
            ax.plot(ks, qs, marker = marker, color = color, alpha = alpha, markersize = markersize)
            
            #ax.plot([np.mean(ks)], [np.mean(qs)], marker = marker, color = color, alpha = alpha, markersize = markersize)
            
            """
            if legend_added:
            #color = np.random.uniform(size = (3,))
                #
                ax.plot([np.mean(ks)], [np.mean(qs)], marker = marker, color = color, alpha = alpha, label = 'mpr = ' + str(mpr), markersize = markersize)
                #
                legend_added = False
            else:
                ax.plot([np.mean(ks)], [np.mean(qs)], marker = marker, color = color, alpha = alpha, markersize = markersize)
            
            """
            
            

        return ax


    @classmethod
    def plot_FD_singlelane_QV(self, FD_singlelane, ax = False, figsize = (5,3), alpha = .4, markersize = 10, color = False, marker = 's', only_use_later_percentage = False):
        """
        
        
        @input: FD_singlelane
            
            FD_singlelane[]
        
            
            reload(CAF)
            #FD_singlelane[desireddensity] = (Q,V,K), Q V K are all list. 
            FD_singlelane = pickle.load(open(dataspath + 'RES_FD_singlelane.pickle', 'rb'))
        
        
        
        @input: only_use_later_percentage
        
            a float within (0, 1)
            
            
        
        
        """
        #
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)#.add_subplot(projection='3d')
        #
        if isinstance(color, bool):
            color = np.random.uniform(size = (3,))
        for desired_density in FD_singlelane.keys():
            #qs = [q1,q2,...qn]
            if isinstance(only_use_later_percentage, bool):
                qs,vs,ks = FD_singlelane[desired_density]
            else:
                qs0,vs0,ks0 = FD_singlelane[desired_density]
                #
                idx = int(len(qs0)*(1-only_use_later_percentage))
                #
                qs = qs0[idx:]
                vs = vs0[idx:]
                ks = ks0[idx:]
                
            
            ax.plot( [np.mean(qs)], [np.mean(vs)],marker = marker, color = color, alpha = alpha, markersize = markersize)
            
            """
            if legend_added:
            #color = np.random.uniform(size = (3,))
                #
                ax.plot([np.mean(ks)], [np.mean(qs)], marker = marker, color = color, alpha = alpha, label = 'mpr = ' + str(mpr), markersize = markersize)
                #
                legend_added = False
            else:
                ax.plot([np.mean(ks)], [np.mean(qs)], marker = marker, color = color, alpha = alpha, markersize = markersize)
            
            """
            
            

        return ax

    @classmethod
    def plot_FD_singlelane(self, FD_singlelane, ax = False, figsize = (5,3), alpha = .4, markersize = 10, color = False, marker = 's', only_use_later_percentage = False):
        """
        
        
        @input: FD_singlelane
            
            FD_singlelane[]
        
            
            reload(CAF)
            #FD_singlelane[desireddensity] = (Q,V,K), Q V K are all list. 
            FD_singlelane = pickle.load(open(dataspath + 'RES_FD_singlelane.pickle', 'rb'))
        
        
        
        @input: only_use_later_percentage
        
            a float within (0, 1)
            
            
        
        
        """
        #
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)#.add_subplot(projection='3d')
        #
        if isinstance(color, bool):
            color = np.random.uniform(size = (3,))
        for desired_density in FD_singlelane.keys():
            #qs = [q1,q2,...qn]
            if isinstance(only_use_later_percentage, bool):
                qs,vs,ks = FD_singlelane[desired_density]
            else:
                qs0,vs0,ks0 = FD_singlelane[desired_density]
                #
                idx = int(len(qs0)*(1-only_use_later_percentage))
                #
                qs = qs0[idx:]
                vs = vs0[idx:]
                ks = ks0[idx:]
                
            
            ax.plot([np.mean(ks)], [np.mean(qs)], marker = marker, color = color, alpha = alpha, markersize = markersize)
            
            """
            if legend_added:
            #color = np.random.uniform(size = (3,))
                #
                ax.plot([np.mean(ks)], [np.mean(qs)], marker = marker, color = color, alpha = alpha, label = 'mpr = ' + str(mpr), markersize = markersize)
                #
                legend_added = False
            else:
                ax.plot([np.mean(ks)], [np.mean(qs)], marker = marker, color = color, alpha = alpha, markersize = markersize)
            
            """
            
            

        return ax

    
    @classmethod
    def plot_capacity_lanesnumber_mpr_askey(self, capacity, ax = False, figsize = (5,3), alpha = .4, markersize = 10, color = False):
        """
        
        @input: capacity
        
            capacity[lanesnumber][mpr] = float, the capacity, unit is veh/h. 
        
        @
        
        """
        #
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)#.add_subplot(projection='3d')
            
            #ax = plt.figure().add_subplot(projection='3d')
            #ax.set_xlabel('t');ax.set_ylabel('x');ax.set_zlabel('speed'); 
            #ax.grid();
            #ax = host_subplot(111)
        
        #
        for ln in capacity.keys():
            mprs = sorted(capacity[ln].keys())
            CAPs = [capacity[ln][mpr] for mpr in mprs]
            ax.plot(mprs, CAPs, label = 'lanes number ' + str(ln))
            
            pass
        ax.set_xlabel('MPR (*100%)');ax.set_ylabel('capacity (veh/h)');
        ax.grid('on')
        ax.legend()
        
        return ax
        pass
    

    
    @classmethod
    def plot_FDs_given_mprs_results_given_lanes_number_averageoverlane_changemarker(self, FD_mprs_given_lanesnumber, ax = False, figsize = (5,3), alpha = .4, markersize = 10, color = False):
        """
        Note that this method just change the marker, not the coloer. 
        
        If want to change the color, using :
        
            - self.plot_FDs_given_mprs_results_given_lanes_number_averageoverlane_changecolor()
        ---------------------------------------------------------
        @input: FD_mprs_given_lanesnumber
        
            FD_mprs_given_lanesnumber[mpr][desired_density] = (Q,K,V)
            
            Q is a dict, Q[laneid] = [q1,q2,...qn], eqch qi is a float. 
        
            It is obtained via: 
                #FD_mprs[lanesnumber][mpr][desired_density] = (Q,K,V)
                #Q is a dict, Q[laneid] = [q1,q2,...qn], eqch qi is a float. 
                FD_mprs = pickle.load(open(dataspath + 'RES_FD_mprs.pickle', 'rb'))
            
            and then FD_mprs_given_lanesnumber = FD_mprs[lanesnumber]
            
        

        
        
        """
        #
        if isinstance(color, bool):
            color = np.random.uniform(size = (3,))
            #marker = np.random.choice(['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_', 'P', 'X', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,])
        """
        if isinstance(marker, bool):
            marker = np.random.choice(['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_', 'P', 'X', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,])
        """
        #
        #
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)#.add_subplot(projection='3d')
            
            #ax = plt.figure().add_subplot(projection='3d')
            #ax.set_xlabel('t');ax.set_ylabel('x');ax.set_zlabel('speed'); 
            #ax.grid();
            #ax = host_subplot(111)
        #
        
        
        for mpr in FD_mprs_given_lanesnumber.keys():
            #
            legend_added = True
            #
            marker = np.random.choice(['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_', 'P', 'X'])
            #
            for desired_density in FD_mprs_given_lanesnumber[mpr].keys():
                #Q[laneid] = [q1,q2,...qn]
                Q,K,V = FD_mprs_given_lanesnumber[mpr][desired_density]
                #
                
                ks = [np.mean(K[laneid]) for laneid in sorted(Q.keys())]
                qs = [np.mean(Q[laneid]) for laneid in sorted(Q.keys())]

                if legend_added:
                #color = np.random.uniform(size = (3,))
                    #
                    ax.plot([np.mean(ks)], [np.mean(qs)], marker = marker, color = color, alpha = alpha, label = 'mpr = ' + str(mpr), markersize = markersize)
                    #
                    legend_added = False
                else:
                    ax.plot([np.mean(ks)], [np.mean(qs)], marker = marker, color = color, alpha = alpha, markersize = markersize)
                    
                    
                    
                    #ax.plot([np.mean(K[laneid])], [np.mean(Q[laneid])],'.', alpha = alpha)
                    
                    """
                    if legend_added:
                    #color = np.random.uniform(size = (3,))
                        #
                        ax.plot([np.mean(K[laneid])], [np.mean(Q[laneid])],'.', color = color, alpha = alpha, label = label)
                        #
                        legend_added = False
                    else:
                        ax.plot([np.mean(K[laneid])], [np.mean(Q[laneid])],'.', color = color, alpha = alpha)
                    """
                
            
            #
        return ax
    
    
    @classmethod
    def plot_FDs_given_mprs_results_given_lanes_number(self, FD_mprs_given_lanesnumber, ax = False, figsize = (5,3), alpha = .4, markersize = 10):
        """
        
        
        @input: FD_mprs_given_lanesnumber
        
            FD_mprs_given_lanesnumber[mpr][desired_density] = (Q,K,V)
            
            Q is a dict, Q[laneid] = [q1,q2,...qn], eqch qi is a float. 
        
            It is obtained via: 
                #FD_mprs[lanesnumber][mpr][desired_density] = (Q,K,V)
                #Q is a dict, Q[laneid] = [q1,q2,...qn], eqch qi is a float. 
                FD_mprs = pickle.load(open(dataspath + 'RES_FD_mprs.pickle', 'rb'))
            
            and then FD_mprs_given_lanesnumber = FD_mprs[lanesnumber]
            
        

        
        
        """
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)#.add_subplot(projection='3d')
            
            #ax = plt.figure().add_subplot(projection='3d')
            #ax.set_xlabel('t');ax.set_ylabel('x');ax.set_zlabel('speed'); 
            #ax.grid();
            #ax = host_subplot(111)
        #
        
        
        for mpr in FD_mprs_given_lanesnumber.keys():
            #
            legend_added = True
            color = np.random.uniform(size = (3,))
            #
            for desired_density in FD_mprs_given_lanesnumber[mpr].keys():
                #Q[laneid] = [q1,q2,...qn]
                Q,K,V = FD_mprs_given_lanesnumber[mpr][desired_density]
                #
                for laneid in Q.keys():
                    #
                    if legend_added:
                    #color = np.random.uniform(size = (3,))
                        #
                        ax.plot([np.mean(K[laneid])], [np.mean(Q[laneid])],'^-', color = color, alpha = alpha, label = 'mpr = ' + str(mpr), markersize = markersize)
                        #
                        legend_added = False
                    else:
                        ax.plot([np.mean(K[laneid])], [np.mean(Q[laneid])],'.', color = color, alpha = alpha,markersize = markersize)
                    
                    
                    
                    #ax.plot([np.mean(K[laneid])], [np.mean(Q[laneid])],'.', alpha = alpha)
                    
                    """
                    if legend_added:
                    #color = np.random.uniform(size = (3,))
                        #
                        ax.plot([np.mean(K[laneid])], [np.mean(Q[laneid])],'.', color = color, alpha = alpha, label = label)
                        #
                        legend_added = False
                    else:
                        ax.plot([np.mean(K[laneid])], [np.mean(Q[laneid])],'.', color = color, alpha = alpha)
                    """
                
            
            #
        return ax
        





class DataAnalysis():
    """
    
    """
    @classmethod
    def AccelerationsFromSimPaths(self, sim_paths, v_idx = 1):
        """
        ACCEs = DataAnalysis.AccelerationsFromSimPaths(sim_paths)
        --------------------
        calculate the jerk from the simulation paths. 
        @input: v_idx
        
            the index of the speed.
        
        @input: sim_paths
        
            sim_paths[veh_idx][path_idx] = pd, columns are moments, which is the same as input arg ts. 
        
        @OUTPUT: ACCEs
            
            ACCEs[veh_idx] is a pd, rows are the path_idx, and columns are ts. 
        
        """
        #jerks[veh_idx]
        ACCEs = {}
        #
        for veh_idx in sim_paths.keys():
            ACCEs_paths = []
            for path_idx in sim_paths[veh_idx].keys():
                #
                ts = sim_paths[veh_idx][path_idx].columns
                deltav = np.diff(sim_paths[veh_idx][path_idx].loc[v_idx,:].values)
                
                acces = deltav/np.diff(ts)
                #
                #---------------------
                ACCEs_paths.append(acces)
            #
            ACCEs[veh_idx] = pd.DataFrame(ACCEs_paths, columns = ts[1:])
        
        return ACCEs
    
    @classmethod
    def BKP_AcceFromSimPaths(self, sim_paths, percentage_used = .4):
        """
        Jerks = DataAnalysis.JerkFromSimPaths(sim_paths)
        --------------------
        calculate the jerk from the simulation paths. 
        
        sim_paths[veh_idx] is a dataframe. shape is (6, len(ts))
        
        @input: sim_paths
        
            sim_paths[veh_idx][path_idx] = pd, columns are moments, which is the same as input arg ts. 
        
        @OUTPUT: jerks
            
            jerks[veh_idx] is a pd, rows are the path_idx, and columns are ts. 
        
        """
        #jerks[veh_idx]
        Acces = {}
        #
        for veh_idx in sim_paths.keys():
            Acces_paths = []
            for path_idx in sim_paths[veh_idx].keys():
                #
                ts = sim_paths[veh_idx][path_idx].columns
                
                #
                jerks = []
                for t0,t1,t2 in zip(ts[:-2], ts[1:-1],ts[2:]):
                    #
                    deltat1 = t1 - t0
                    deltat2 = t2 - t1
                    #
                    deltav1 = sim_paths[veh_idx][path_idx].loc[1, t1] - sim_paths[veh_idx][path_idx].loc[1, t0]
                    deltav2 = sim_paths[veh_idx][path_idx].loc[1, t2] - sim_paths[veh_idx][path_idx].loc[1, t1]
                    #
                    jerk = (deltav1/deltat1 - deltav2/deltat2)/deltat2
                    #(sim_paths[veh_idx][path_idx].loc[1, t2] + sim_paths[veh_idx][path_idx].loc[1, t0] - 2.0*sim_paths[veh_idx][path_idx].loc[1, t1])
                    jerks.append(jerk)
                #---------------------
                jerks_paths.append(jerks)
            #
            Jerks[veh_idx] = pd.DataFrame(jerks_paths, columns = ts[2:])
        
        return Jerks
    
    
    @classmethod
    def JerkFromSimPaths(self, sim_paths, percentage_used = .4):
        """
        Jerks = DataAnalysis.JerkFromSimPaths(sim_paths)
        --------------------
        calculate the jerk from the simulation paths. 
        
        sim_paths[veh_idx] is a dataframe. shape is (6, len(ts))
        
        @input: sim_paths
        
            sim_paths[veh_idx][path_idx] = pd, columns are moments, which is the same as input arg ts. 
        
        @OUTPUT: jerks
            
            jerks[veh_idx] is a pd, rows are the path_idx, and columns are ts. 
        
        """
        #jerks[veh_idx]
        Jerks = {}
        #
        for veh_idx in sim_paths.keys():
            jerks_paths = []
            for path_idx in sim_paths[veh_idx].keys():
                #
                ts = sim_paths[veh_idx][path_idx].columns
                #
                jerks = []
                for t0,t1,t2 in zip(ts[:-2], ts[1:-1],ts[2:]):
                    #
                    deltat1 = t1 - t0
                    deltat2 = t2 - t1
                    #
                    deltav1 = sim_paths[veh_idx][path_idx].loc[1, t1] - sim_paths[veh_idx][path_idx].loc[1, t0]
                    deltav2 = sim_paths[veh_idx][path_idx].loc[1, t2] - sim_paths[veh_idx][path_idx].loc[1, t1]
                    #
                    jerk = (deltav1/deltat1 - deltav2/deltat2)/deltat2
                    #(sim_paths[veh_idx][path_idx].loc[1, t2] + sim_paths[veh_idx][path_idx].loc[1, t0] - 2.0*sim_paths[veh_idx][path_idx].loc[1, t1])
                    jerks.append(jerk)
                #---------------------
                jerks_paths.append(jerks)
            #
            Jerks[veh_idx] = pd.DataFrame(jerks_paths, columns = ts[2:])
        
        return Jerks
    
    @classmethod
    def FilterOutNanInf(self, data, anotherdata):
        data = np.array(data)
        anotherdata = np.array(anotherdata)
        #
        idx = np.isnan(data)
        #print(idx, type(idx))
        data0 = data[~idx]
        anotherdata0 = anotherdata[~idx]
        #
        idx = np.isinf(data0)
        
        data1 = data0[~idx]
        anotherdata1 = anotherdata0[~idx]
        return data1,anotherdata1


    
    @classmethod
    def JerkInterTime_percentile(self, data_pd, jerk_percentile, vids_col = 'id', longitudinalspeedcol = 'xVelocity', col = 'xAcceleration', timecol  = 'frame', dt = 0.1, convert2meter = .3028):
        """
        @input: col
        
            the name of the columns of acceleration.
            
            It is used to calculate the jerk. 
        
        
        durations = 
        """
        #data_pd['jerk'] = 
        #
        ##################################
        durations = []
        vids = set(data_pd[vids_col])
        for vid in vids:
            #vdata
            vdata = data_pd[data_pd[vids_col]==vid]
            sorted_df = vdata.sort_values(by=timecol)
            #data is too less to calculate the jerk. 
            if sorted_df.shape[0]<=2:continue
            #print(sorted_df.shape)
            #JERKS is a array. 
            JERKS = convert2meter*np.diff(sorted_df[col].values[:])/(dt*np.diff(sorted_df[timecol].values[:]))
            #print(max(JERKS))
            if jerk_percentile>0:
                indexes = np.where(JERKS>=jerk_percentile)[0]
            elif jerk_percentile<0:
                indexes = np.where(JERKS<=jerk_percentile)[0]
            if len(indexes)<=1:continue
            durations.extend(dt*np.diff(sorted_df[timecol].iloc[indexes]))
        
        return durations


    @classmethod
    def JerkFromData_using_3rddifferential(self, data_pd, longitudinalspeedcol = 'xVelocity', loc_col = 'Local_Y', timecol  = 'frame', dt = 0.1, convert2meter = .3028, vids_col = 'id', SPEEDMAX_ms = False, ACCEMAX_MS2 = False):
        """
        Calculate the jerk from the data. 
        
        Difference:
        
            self.JerkFromData
            self.JerkFromData_using_2nddifferential
            
        The former one using acceleration, while the latter one using the location, differential three times. 
        
        ----------------------------------------------
        @input: dp
        
            pandas data frame. 
            
        @input: col
        
            the name of the columns that indicate the acceleration. 
        
        @input: longitudinalspeedcol
        
            the name of the column that indicate the 
        
        @OUTPUT: speedvsjerks
        
            two list with the same length. 
            
        """
        #
        SPEEDS = []
        JERKS = []
        #sort the dataframe based on col timecol
        # Sort the rows based on the 'Score' column
        #
        #acces = sorted_df[xAcceleration].values[1:]
        vids = set(data_pd[vids_col])
        for vid in vids:
            #vdata
            vdata = data_pd[data_pd[vids_col]==vid]
            sorted_df = vdata.sort_values(by=timecol)
            #data is too less to calculate the jerk. 
            if sorted_df.shape[0]<=3:continue
            #
            speeds = convert2meter*abs(sorted_df[longitudinalspeedcol].values[3:])
            #-------First differential, unit is m/s 
            Differential_1st = convert2meter*np.diff(sorted_df[loc_col].values[:])/(dt*np.diff(sorted_df[timecol].values[:]))
            #speeds = convert2meter*abs(sorted_df[longitudinalspeedcol].values[1:])
            #times4delta = sorted_df[timecol].values[1:]
            #
            #Second differential, unit is m/s^2
            Differential_2nd = np.diff(Differential_1st)/(dt*np.diff(sorted_df[timecol].values[1:]))
            Differential_3rd = np.diff(Differential_2nd)/(dt*np.diff(sorted_df[timecol].values[2:]))
            #
            #print(len(Differential_3rd), len(speeds))
            JERKS.extend(Differential_3rd)
            SPEEDS.extend(speeds)
        #
        return SPEEDS,JERKS




    @classmethod
    def JerkFromData_using_3rddifferential_tmp(self, data_pd, longitudinalspeedcol = 'xVelocity', loc_col = 'Local_Y', timecol  = 'frame', dt = 0.1, convert2meter = .3028, vids_col = 'id', SPEEDMAX_ms = False, ACCEMAX_MS2 = False):
        """
        Calculate the jerk from the data. 
        
        Difference:
        
            self.JerkFromData
            self.JerkFromData_using_2nddifferential
            
        The former one using acceleration, while the latter one using the location, differential three times. 
        
        ----------------------------------------------
        @input: dp
        
            pandas data frame. 
            
        @input: col
        
            the name of the columns that indicate the acceleration. 
        
        @input: longitudinalspeedcol
        
            the name of the column that indicate the 
        
        @OUTPUT: speedvsjerks
        
            two list with the same length. 
            
        """
        #
        SPEEDS = []
        JERKS = []
        #sort the dataframe based on col timecol
        # Sort the rows based on the 'Score' column
        #
        #acces = sorted_df[xAcceleration].values[1:]
        vids = set(data_pd[vids_col])
        for vid in vids:
            #vdata
            vdata = data_pd[data_pd[vids_col]==vid]
            sorted_df = vdata.sort_values(by=timecol)
            #data is too less to calculate the jerk. 
            if sorted_df.shape[0]<=2:continue
            #
            #-------First differential, unit is m/s 
            if not isinstance(SPEEDMAX_ms, bool):
                Differential_1st = convert2meter*np.diff(sorted_df[loc_col].values[:])/(dt*np.diff(sorted_df[timecol].values[:]))
                speeds = convert2meter*abs(sorted_df[longitudinalspeedcol].values[1:])
                times4delta = sorted_df[timecol].values[1:]
                #
                valid_indexs = np.where(Differential_1st<=SPEEDMAX_ms)[0]
                #Differential_1st[Differential_1st>SPEEDMAX_ms] = SPEEDMAX_ms
                #Differential_1st[Differential_1st<0] = 0
                Differential_1st  = Differential_1st[valid_indexs]
                speeds = speeds[valid_indexs]
                times4delta = times4delta[valid_indexs]
                #print(len(valid_indexs), len(times4delta), len(Differential_1st), sorted_df.shape)
            else:
                
                Differential_1st = convert2meter*np.diff(sorted_df[loc_col].values[:])/(dt*np.diff(sorted_df[timecol].values[:]))
                speeds = convert2meter*abs(sorted_df[longitudinalspeedcol].values[1:])
                times4delta = sorted_df[timecol].values[1:]
            #
            if not isinstance(ACCEMAX_MS2, bool):
                #
                #Second differential, unit is m/s^2
                Differential_2nd = np.diff(Differential_1st)/(dt*np.diff(times4delta))
                speeds = speeds[1:]
                times4delta = times4delta[1:]
                #
                valid_indexs = np.where(Differential_2nd<=ACCEMAX_MS2)[0]
                
                Differential_2nd = Differential_2nd[valid_indexs]
                speeds = speeds[valid_indexs]
                times4delta = times4delta[valid_indexs]
                
                #Differential_2nd[Differential_2nd>ACCEMAX_MS2] = ACCEMAX_MS2
                #Differential_2nd[Differential_2nd<-ACCEMAX_MS2] = -ACCEMAX_MS2
            else:
                Differential_2nd = np.diff(Differential_1st)/(dt*np.diff(times4delta))
                speeds = speeds[1:]
                times4delta = times4delta[1:]
            #print(max(Differential_2nd  ))
            #third differential, unit is m/s^3
            Differential_3rd = np.diff(Differential_2nd)/(dt*np.diff(times4delta))
            #
            #print(len(Differential_3rd), len(speeds))
            JERKS.extend(Differential_3rd)
            SPEEDS.extend(speeds[1:])
        #
        return SPEEDS,JERKS


    @classmethod
    def JerkFromData(self, data_pd, longitudinalspeedcol = 'xVelocity', col = 'xAcceleration', timecol  = 'frame', dt = 0.1, convert2meter = .3028, vids_col = 'id'):
        """
        Calculate the jerk from the data. 
        
        @input: dp
        
            pandas data frame. 
            
        @input: col
        
            the name of the columns that indicate the acceleration. 
        
        @input: longitudinalspeedcol
        
            the name of the column that indicate the 
        
        @OUTPUT: speedvsjerks
        
            two list with the same length. 
            
        """
        #
        SPEEDS = []
        JERKS = []
        #sort the dataframe based on col timecol
        # Sort the rows based on the 'Score' column
        #
        #acces = sorted_df[xAcceleration].values[1:]
        vids = set(data_pd[vids_col])
        for vid in vids:
            #vdata
            vdata = data_pd[data_pd[vids_col]==vid]
            sorted_df = vdata.sort_values(by=timecol)
            #data is too less to calculate the jerk. 
            if sorted_df.shape[0]<=2:continue
            #
            SPEEDS.extend(convert2meter*abs(sorted_df[longitudinalspeedcol].values[1:]))
            #
            JERKS.extend(convert2meter*np.diff(sorted_df[col].values[:])/(dt*np.diff(sorted_df[timecol].values[:])))
            
        return SPEEDS,JERKS
        
        pass
    
    @classmethod
    def Volatility_pd(self, data_pd, latter_pencentage = 0.5, ):
        """
        THe volatility for one series. 
        
        @input: data_pd
        
            a pd. The columns are the moments and the row is [x, vx, y, vy, zlon, zlat]
        
        @input: latter_pencentage
        
            a float within [0, 1].
            
            
        @OUTPUT: volatilities
        
            a dict. 
            
            keys include:
            
                'jerk':
                    'coeffi', which is calculated as sigma/mean
                
                    'std'
                    
                'acceleration'
            
        """
        #
        volatilities = {'jerk':{}, 'acceleration':{}}
        #
        #######################Jerk
        ts = data_pd.columns
        #
        jerks = []
        for t0,t1,t2 in zip(ts[:-2], ts[1:-1],ts[2:]):
            #
            deltat1 = t1 - t0
            deltat2 = t2 - t1
            #
            deltav1 = sim_paths[veh_idx][path_idx].loc[1, t1] - sim_paths[veh_idx][path_idx].loc[1, t0]
            deltav2 = sim_paths[veh_idx][path_idx].loc[1, t2] - sim_paths[veh_idx][path_idx].loc[1, t1]
            #
            jerk = (deltav1/deltat1 - deltav2/deltat2)/deltat2
            #(sim_paths[veh_idx][path_idx].loc[1, t2] + sim_paths[veh_idx][path_idx].loc[1, t0] - 2.0*sim_paths[veh_idx][path_idx].loc[1, t1])
            jerks.append(jerk)
        #
        
        #######################Acce
        ts = data_pd.columns
        deltav = np.diff(data_pd.loc[1,:].values)
        acces = deltav/np.diff(ts)
        #
        
        
        
        pass
    
    @classmethod
    def LateralDynamicsExtractGivenSimResOnClosedRoad(self, STATES, irregular_vehs_ids = [], filter_max_acce_jerk = False, jerkMAX = 5, typee = 'all'):
        """
        Extract the lateral dynamics from the simulation results of the 
        
        ----------------------------------
        @input: typee
        
            the typee of the processing of the data. 
            
            If it is:
            
                'all': take account all vehicles
                
                'only_irregular': only vehicles in list irregular_vehs_ids. 
        
                'only_non_irregular': only vehicles not in list irregular_vehs_ids. 
        
        @input: STATES,irregular_vehs_ids
            
            STATES is a dict. 
                
                STATES[moment][vid] is a 1d array, shape is (6,), they are:
                    
                    self.vehs_dict[vehicle_id] = [x, vx, y, vy, zlon, zlat]
            
            irregular_vehs_ids is a list, containing the vehicle ids that have jump terms. 
            
            Can be obtained via the following:
            
                --------irregular case
                    reload(irre)
                    T_horizon_sec = 400
                    ts = np.linspace(0, T_horizon_sec, int(T_horizon_sec/.4))
                    #==================Number of irregular vehicles. 
                    #------------------irregular vehicle args
                    N_irregulars = 0
                    args_jumpsize_diss = [{'mu':.0, 'sigma': 1.0} for i in range(N_irregulars)]
                    jumpdensities =  [1 for i in range(N_irregulars)]
                    jumpdis_types =  ['Normal' for i in range(N_irregulars)]
                    #------------------irregular vehicle jump times and jump size. 
                    #jumptimes_es_lon_es[veh_idx][path_idx] = [t1, t2, ....tn]
                    #jumpsizes_es_lon_es[veh_idx][path_idx] = [js1, js2,....]
                    jumptimes_es_lon_es,jumpsizes_es_lon_es = irre.JumpDiffusionDrivingBehavior.JumpMomentsWithJumpSizes_multiplevehicles(\
                                T_horizon_sec= T_horizon_sec, jumpdis_types = jumpdis_types, \
                                args_jumpsize_diss = args_jumpsize_diss, jumpdensities = jumpdensities, N_paths  = 1)
                    jumptimes_es_lat_es,jumpsizes_es_lat_es = irre.JumpDiffusionDrivingBehavior.JumpMomentsWithJumpSizes_multiplevehicles(\
                                T_horizon_sec= T_horizon_sec, jumpdis_types = jumpdis_types, \
                                args_jumpsize_diss = args_jumpsize_diss, jumpdensities = jumpdensities, N_paths  = 1)
                    #, jump_or_nots = jump_or_nots_vehs_paths[veh_idx][path_idx], is a list of True or False
                    #jumpsizes_adapted = jumpsizes_adapted_vehs_paths[veh_idx][path_idx],is either 0 or a jumptize
                    adapted_ts_list,jump_or_nots_lists,jumpsizes_adapted_lists = irre.JumpDiffusionDrivingBehavior.AdaptEuler(ts = ts, \
                                                    jumptimes_vehs_paths = jumptimes_es_es, jumpsizes_vehs_paths = jumpsizes_es_es, )
                    #####################################SIMULATION AND PLOT
                    reload(irre)
                    two_dim_paras = {'alpha_roadbounds': 1.0 , 'beta_lane_marks':3.6, 'beta_lane_middle_line': 1.0, \
                        'sigma_long': .5, 'sigma_lat':.2, 'sigma_long_drift':1.0, 'sigma_lat_drift':1.0, 'theta_ou':.1, \
                                'amplyfier_bound': 1.0, 'amplyfier_lane_mark': 1.0,  'amplyfier_intra_lanes_long': 1e-2, 'amplyfier_intra_lanes_lat':1e-2}
                    idm_paras = {'idm_vf':10.0/3.6, 'idm_T':1.5, 'idm_delta':4.0, 'idm_s0':2.0, 'idm_a':1.0, 'idm_b':3.21, 'a_MAX':3.5, 'veh_len':4}
                    #
                    road = irre.SingleLaneSim()
                    STATES,irregular_vehs_ids = road.sim_insert_all_vehs_at_once_jump(adapted_ts = adapted_ts, \
                                    jumpsizes_adapted_lists = jumpsizes_adapted_lists, desired_density = 0, idm_paras = idm_paras, \
                                    two_dim_paras = two_dim_paras, two_dim_paras_irregular = two_dim_paras, jump_or_not = True, delay_tao_sec = .4)
                    #
                    ax = road.plot_lanemarks_boundary(figsize = (4, 3), markcolor = 'y')
                    #fig,ax = plt.subplots(figsize = (10, 4), )
                    ax = irre.SingleLaneSim.plot_sim_from_snapshots(snapshots = road.snapshots, \
                               figsize = (7, 7),  ax = ax, alpha = 1, )
                
                -----------------Regular case. 
        
        @OUTPUT: res
            
            Res['speed']  is a list. 
            res['acceleration']  is a list. 
            res['jerk']  is a list. 
            res['speed_vs_yjerk']  is a dict. 
            res['speed_vs_xjerk']  is a dict. 
            
        """
        #
        Res = {'speed':[], 'acceleration':[], 'jerk':[], 'speed_vs_yjerk':{'speeds':[], 'jerks':[]},'speed_vs_xjerk':{'speeds':[], 'jerks':[]} }
        #STATES[moment][vid] is a 1d array, shape is (6,)
        ts = sorted(STATES.keys())
        #speed
        for t0,t1,t2 in zip(ts[:-2], ts[1:-1], ts[2:]):
            deltat1 = t1 - t0
            deltat2 = t2 - t1
            #
            vids = STATES[t0].keys()
            #
            for vid in vids:
                #-------------------------------
                if typee=='only_irregular':
                    if vid not in irregular_vehs_ids:
                        continue
                elif typee=='only_non_irregular':
                    if vid in irregular_vehs_ids:
                        continue
                #-------------------------------
                #x, vx, y, vy, zlon, zlat = STATES[t][vid]
                #x, vx, y, vy, zlon, zlat = STATES[t][vid]
                #
                Res['speed'].append(STATES[t0][vid][3])
                Res['speed'].append(STATES[t1][vid][3])
                Res['speed'].append(STATES[t2][vid][3])
                #
                Res['acceleration'].append((STATES[t1][vid][3]  -STATES[t0][vid][3])/deltat1)
                Res['acceleration'].append((STATES[t2][vid][3]  -STATES[t1][vid][3])/deltat2)
                #
                jerk = ((STATES[t2][vid][3]  -STATES[t1][vid][3])/deltat2 - (STATES[t1][vid][3]  -STATES[t0][vid][3])/deltat1)/deltat2
                Res['jerk'].append(jerk)
                #speed_vs_yjerk
                if filter_max_acce_jerk:
                    if jerk>-jerkMAX and jerk<jerkMAX and STATES[t1][vid][1]>=0:
                        Res['speed_vs_yjerk']['speeds'].append(STATES[t1][vid][1])
                        Res['speed_vs_yjerk']['jerks'].append(jerk)
                else:
                    Res['speed_vs_yjerk']['speeds'].append(STATES[t1][vid][3])
                    Res['speed_vs_yjerk']['jerks'].append(jerk)
                #speed_vs_xjerk
                jerk = ((STATES[t2][vid][1]  -STATES[t1][vid][1])/deltat2 - (STATES[t1][vid][1]  -STATES[t0][vid][1])/deltat1)/deltat2
                if filter_max_acce_jerk:
                    if jerk>-jerkMAX and jerk<jerkMAX and STATES[t1][vid][1]>=0:
                        Res['speed_vs_xjerk']['speeds'].append(STATES[t1][vid][1])
                        Res['speed_vs_xjerk']['jerks'].append(jerk)
                else:
                    Res['speed_vs_xjerk']['speeds'].append(STATES[t1][vid][3])
                    Res['speed_vs_xjerk']['jerks'].append(jerk)

        return Res
        

    
    @classmethod
    def lateraldynamics_extract(self, paired_data_pd, delay_n = 0, distance_threshold  = 3.5):
        """
        From the paired data to extract the lateral dynamics. 
        
        The data describe the dynamics of the ego vehicle, ego lane leader, and the vehicle at neighboring lane (which is ahead of the ego vehicle). 
        
        #
                        
            reload(irre)
            res = irre.DataAnalysis.lateraldynamics_extract(data)
            #
            fig,ax = plt.subplots()
            ax.plot(res['distance_vs_ego_acceleration']['dis'], res['distance_vs_ego_acceleration']['ego_accelerations_y'],'.')
            
        ------------------------------------------------------------------
        @input: delay_n
        
            the delay of the time. 
            
        @input: paired_data_pd
        
            a dataframe that describe the ego data, 
        
            Index(['frame', 'ego_id', 'fr_id', 'fl_id', 'ego_x', 'ego_y', 'ego_xVelocity',
                   'ego_yVelocity', 'ego_xAcceleration', 'ego_yAcceleration', 'ego_laneId',
                   'fr_x', 'fr_y', 'fr_xVelocity', 'fr_yVelocity', 'fr_xAcceleration',
                   'fr_yAcceleration', 'fr_laneId', 'fl_x', 'fl_y', 'fl_xVelocity',
                   'fl_yVelocity', 'fl_xAcceleration', 'fl_yAcceleration', 'fl_laneId'],
                  dtype='object')
        
            x is longitudinal
            y is lateral
            
            fl is front-left
            fr is front-right
        
        @OUTPUT: res
        
            res['distance_vs_ego_acceleration'] is the relationship between the distance of the two vehicles and the acceleration of the ego vehic.e
            
            res['distance_vs_ego_acceleration']['distance_between_two_vehs']
        
        """
        #
        res = {}
        #
        #==============================
        res['jerks'] = []
        #total_acceleration is 
        res['distance_vs_ego_acceleration'] = {'dis':[], 'ego_accelerations_y':[], 'ego_yVelocity':[], 'relative_v':[], 'relative_acceleration':[], }
        #
        distance = paired_data_pd['ego_y']
        ego_ids = set(paired_data_pd['ego_id'])
        for ego_id in ego_ids:
            #
            #Jerks
            veh_ego_data = paired_data_pd[paired_data_pd['ego_id'] == ego_id]
            #
            ##################Left lane, or fl
            fl_ids = set(veh_ego_data.fl_id)
            for fl_id in fl_ids:
                #
                if np.isnan(fl_id):continue
                #
                scenariodata = veh_ego_data[veh_ego_data.fl_id == fl_id]
                #   distances
                distances  = abs(scenariodata.ego_y - scenariodata.fl_y).values
                if scenariodata.ego_y.iloc[0] - scenariodata.fl_y.iloc[0]>0:
                    direction = -1
                    pass
                else:
                    direction = 1
                
                """
                #   acceleration
                delta_vys = scenariodata.fl_yVelocity.values[1:] - scenariodata.fl_yVelocity.values[:-1]
                delta_ts = scenariodata.frame.values[1:] - scenariodata.frame.values[:-1]
                ego_accelerations_y = delta_vys/delta_ts
                #
                if len(distances)!=len(ego_accelerations_y)+1:
                    print(scenariodata.shape, len(delta_vys), len(distances), len(ego_accelerations_y))
                    raise ValueError('''''')
                """    
                relative_v =  (scenariodata.ego_yVelocity - scenariodata.fl_yVelocity).values
                ego_accelerations_y = scenariodata.ego_yAcceleration.values
                ego_yVelocity = scenariodata.ego_yVelocity.values
                relative_acceleration = (scenariodata.ego_yAcceleration - scenariodata.fl_yAcceleration).values
                #
                idxs = np.where(distances<=distance_threshold)[0]
                tmp_length = len(idxs)
                if len(idxs)<=delay_n:continue
                #print(idxs)
                res['distance_vs_ego_acceleration']['dis'].extend(distances[idxs][:tmp_length-delay_n])
                res['distance_vs_ego_acceleration']['ego_accelerations_y'].extend(direction*ego_accelerations_y[idxs][delay_n:])
                res['distance_vs_ego_acceleration']['ego_yVelocity'].extend(direction*ego_yVelocity[idxs][delay_n:])
                res['distance_vs_ego_acceleration']['relative_v'].extend(relative_v[idxs][delay_n:])
                res['distance_vs_ego_acceleration']['relative_acceleration'].extend(relative_acceleration[idxs][delay_n:])
            ################Right lane
            fr_ids = set(veh_ego_data.fr_id)
            for fr_id in fr_ids:
                #
                if np.isnan(fr_id):continue
                #
                scenariodata = veh_ego_data[veh_ego_data.fr_id == fr_id]
                #   distances
                distances  = abs(scenariodata.ego_y - scenariodata.fr_y).values
                if scenariodata.ego_y.iloc[0] - scenariodata.fr_y.iloc[0]>0:
                    direction = -1
                    pass
                else:
                    direction = 1
                """
                #   acceleration
                #ego_accelerations_y = (scenariodata.fr_yVelocity.iloc[1:] - scenariodata.fr_yVelocity.iloc[:-1])/(scenariodata.frame.iloc[1:] - scenariodata.frame.iloc[:-1])
                delta_vys = scenariodata.fr_yVelocity.values[1:] - scenariodata.fr_yVelocity.values[:-1]
                delta_ts = scenariodata.frame.values[1:] - scenariodata.frame.values[:-1]
                ego_accelerations_y = delta_vys/delta_ts
                """
                relative_v =  (scenariodata.ego_yVelocity - scenariodata.fr_yVelocity).values
                ego_accelerations_y = scenariodata.ego_yAcceleration.values
                ego_yVelocity = scenariodata.ego_yVelocity.values
                relative_acceleration = (scenariodata.ego_yAcceleration - scenariodata.fr_yAcceleration).values
                #
                #idxs = np.array(range(len(distances)))[distances<=3.5]
                idxs = np.where(distances<=distance_threshold)[0]
                tmp_length = len(idxs)
                if len(idxs)<=delay_n:continue
                res['distance_vs_ego_acceleration']['dis'].extend(distances[idxs][:tmp_length-delay_n])
                res['distance_vs_ego_acceleration']['ego_accelerations_y'].extend(direction*ego_accelerations_y[idxs][delay_n:])
                res['distance_vs_ego_acceleration']['ego_yVelocity'].extend(direction*ego_yVelocity[idxs][delay_n:])
                res['distance_vs_ego_acceleration']['relative_v'].extend(relative_v[idxs][delay_n:])
                res['distance_vs_ego_acceleration']['relative_acceleration'].extend(relative_acceleration[idxs][delay_n:])
        #---------------------------------------------------------
        
        return res
    
    


class JumpDiffusionDrivingBehavior():
    """
    The jump-diffusion approach for 
    """
    
    @classmethod
    def AdaptEuler_BKP(slef, ts, jumptimes, jumpsizes):
        """
        adapt the times for jump and diffusion. 
        
        Callback methods:
        
            adapted_ts,jump_or_nots,jumpsizes_adapted = self.AdaptEuler(ts = ts, jumptimes = jumptimes, jumpsizes = jumpsizes)
        
        ---------------------------
        @input: ts
        
            a list. possiblly in equal distance. 
        
        @input: jumptimes
            
            the jump moments generated by self.JumpMomentsWithJumpSizes()
        
        @OUTPUT: adapted_ts,jump_or_nots,jumpsizes_adapted
        
            three lists with the same lenght. 
        
            jump_or_nots is a list of True or False. 
            
            If a element is True, then 
            
            jumpsizes_adapted is either 0 or a jumptize. 
        
        """
        adapted_ts = sorted(set(list(ts) + list(jumptimes)))
        
        idx_jumpsizes = 0
        jump_or_nots = []
        jumpsizes_adapted = []
        for t in adapted_ts:
            if t in jumptimes:
                jump_or_nots.append(True)
                jumpsizes_adapted.append(jumpsizes[idx_jumpsizes])
                idx_jumpsizes = idx_jumpsizes + 1
            else:
                jump_or_nots.append(False)
                jumpsizes_adapted.append(0)
        #
        return adapted_ts,jump_or_nots,jumpsizes_adapted
    
    
    @classmethod
    def AdaptEuler_still_regular(slef, ts, jumptimes_vehs_paths, jumpsizes_vehs_paths):
        """
        adapt the times for jump and diffusion. 
        
        But the returned ts is still regular, which is the same as input arg ts. 
        
        Callback methods:
            
            #
            reload(irre)
            ts = np.linspace(0, 200, 400)
            #
            jumptimes_vehs_paths, jumpsizes_vehs_paths = self.JumpMomentsWithJumpSizes_multiplevehicle(T_horizon_sec = 4000, N_paths = 100)
            #
            adapted_ts,jump_or_nots_vehs_paths,jumpsizes_adapted_vehs_paths = self.AdaptEuler(ts = ts, jumptimes_vehs_paths = jumptimes_vehs_paths, jumpsizes_vehs_paths = jumpsizes_vehs_paths)
        
        ---------------------------
        @input: ts
        
            a list. possiblly in equal distance. 
        
        @input: jumptimes_vehs_paths and jumpsizes_vehs_paths
            
            both are dic of dict. 
            
            jumptimes = jumptimes_es[veh_idx][path_idx]
            jumpsizes = jumpsizes_es[veh_idx][path_idx]
            
            jumptimes and jumpsizes
                
                both are iteratible. 
                
                jumptimes = [t1, t2, t3....], each element is a float.
                jumpsizes = [js1, js2,....], each element is a float. 
                
            They are obtained via:
                jumptimes_vehs_paths, jumpsizes_vehs_paths = self.JumpMomentsWithJumpSizes_multiplevehicle(T_horizon_sec = 4000, N_paths = 100)
            

        
        @OUTPUT: adapted_ts,jump_or_nots_vehs_paths,jumpsizes_adapted_vehs_paths
            
            three lists.
            
            adapted_ts is a list containing the moments.
            
            jump_or_nots = jump_or_nots_vehs_paths[veh_idx][path_idx], is a list of True or False.
                jump_or_nots is a list of True or False. 
            
                If a element is True, then 
            
            jumpsizes_adapted = jumpsizes_adapted_vehs_paths[veh_idx][path_idx],is either 0 or a jumptize. 
                
                jumpsizes_adapted is either 0 or a jumptize. 
        
        """
        
        #DISCARDE: #jump_or_nots_vehs_paths[veh_idx][path_idx]  = [True, False, ...]
        #DISCARDE: jump_or_nots_vehs_paths = {}
        #jumpsizes_adapted_vehs_paths[veh_idx][path_idx]  = [size1, size 2, ...]
        jumpsizes_adapted_vehs_paths = {}
        """
        #------------------------------------------adapted_ts
        adapted_ts0 = list(ts)
        for veh_idx in jumptimes_vehs_paths.keys():
            jumptimes_veh = jumptimes_vehs_paths[veh_idx]
            for path_idx in jumptimes_veh.keys():
                #
                jumptimes = jumptimes_veh[path_idx]
                #
                adapted_ts0.extend(jumptimes)
        adapted_ts = sorted(set(adapted_ts0))
        """
        adapted_ts = ts
        #
        #------------------------------------------jump_or_nots_vehs_paths and jumpsizes_adapted_vehs_paths
        for veh_idx in jumptimes_vehs_paths.keys():
            #RAW data
            jumptimes_veh = jumptimes_vehs_paths[veh_idx]
            jumpsizes_veh = jumpsizes_vehs_paths[veh_idx]
            #
            #DISCARDE: jump_or_nots_vehs = {}
            jumpsizes_adapted_vehs = {}
            #
            for path_idx in jumptimes_veh.keys():
                jumptimes = jumptimes_veh[path_idx]
                jumpsizes = jumpsizes_veh[path_idx]
                #
                #jump_or_nots and jumpsizes_adapted are two lists. 
                #DISCARDE: jump_or_nots = []
                jumpsizes_adapted = []
                #the index used in the jumped series
                idx_adapted = 0
                #idx_regular is used in the regular time grid, or adapted_ts.
                for idx_regular in range(1, len(adapted_ts)):
                    jump_in_interval = .0
                    for idx,(t_jump,size) in enumerate(zip(jumptimes[idx_adapted:], jumpsizes[idx_adapted:])):
                        if t_jump<=adapted_ts[idx_regular] and t_jump>adapted_ts[idx_regular-1]:
                            jump_in_interval = jump_in_interval + size
                            idx_adapted = idx_adapted + 1
                        else:
                            break
                    #
                    jumpsizes_adapted.append(jump_in_interval)
                    
                    """DISCARDE
                    if t in jumptimes:
                        jump_or_nots.append(True)
                        jumpsizes_adapted.append(jumpsizes[idx])
                        idx = idx + 1
                    else:
                        jump_or_nots.append(False)
                        jumpsizes_adapted.append(0)
                    """
                #DISCARDE: jump_or_nots_vehs[path_idx]  = jump_or_nots
                jumpsizes_adapted_vehs[path_idx] = jumpsizes_adapted
            #
            #DISCARDE: jump_or_nots_vehs_paths[veh_idx] = jump_or_nots_vehs
            jumpsizes_adapted_vehs_paths[veh_idx]  = jumpsizes_adapted_vehs
        #
        return adapted_ts,jumpsizes_adapted_vehs_paths
        #
        return adapted_ts,jump_or_nots_vehs_paths,jumpsizes_adapted_vehs_paths
        


    
    @classmethod
    def AdaptEuler(slef, ts, jumptimes_vehs_paths, jumpsizes_vehs_paths):
        """
        adapt the times for jump and diffusion. 
        
        Callback methods:
            
            #
            reload(irre)
            ts = np.linspace(0, 200, 400)
            #
            jumptimes_vehs_paths, jumpsizes_vehs_paths = self.JumpMomentsWithJumpSizes_multiplevehicle(T_horizon_sec = 4000, N_paths = 100)
            #
            adapted_ts,jump_or_nots_vehs_paths,jumpsizes_adapted_vehs_paths = self.AdaptEuler(ts = ts, jumptimes_vehs_paths = jumptimes_vehs_paths, jumpsizes_vehs_paths = jumpsizes_vehs_paths)
        
        ---------------------------
        @input: ts
        
            a list. possiblly in equal distance. 
        
        @input: jumptimes_vehs_paths and jumpsizes_vehs_paths
            
            both are dic of dict. 
            
            jumptimes = jumptimes_es[veh_idx][path_idx]
            jumpsizes = jumpsizes_es[veh_idx][path_idx]
            
            jumptimes and jumpsizes
                
                both are iteratible. 
                
                jumptimes = [t1, t2, t3....], each element is a float.
                jumpsizes = [js1, js2,....], each element is a float. 
                
            They are obtained via:
                jumptimes_vehs_paths, jumpsizes_vehs_paths = self.JumpMomentsWithJumpSizes_multiplevehicle(T_horizon_sec = 4000, N_paths = 100)
            

        
        @OUTPUT: adapted_ts,jump_or_nots_vehs_paths,jumpsizes_adapted_vehs_paths
            
            three lists.
            
            adapted_ts is a list containing the moments.
            
            jump_or_nots = jump_or_nots_vehs_paths[veh_idx][path_idx], is a list of True or False.
                jump_or_nots is a list of True or False. 
            
                If a element is True, then 
            
            jumpsizes_adapted = jumpsizes_adapted_vehs_paths[veh_idx][path_idx],is either 0 or a jumptize. 
                
                jumpsizes_adapted is either 0 or a jumptize. 
        
        """
        
        #jump_or_nots_vehs_paths[veh_idx][path_idx]  = [True, False, ...]
        jump_or_nots_vehs_paths = {}
        #jumpsizes_adapted_vehs_paths[veh_idx][path_idx]  = [size1, size 2, ...]
        jumpsizes_adapted_vehs_paths = {}
        
        #------------------------------------------adapted_ts
        adapted_ts0 = list(ts)
        for veh_idx in jumptimes_vehs_paths.keys():
            jumptimes_veh = jumptimes_vehs_paths[veh_idx]
            for path_idx in jumptimes_veh.keys():
                #
                jumptimes = jumptimes_veh[path_idx]
                #
                adapted_ts0.extend(jumptimes)
        adapted_ts = sorted(set(adapted_ts0))
        #
        #------------------------------------------jump_or_nots_vehs_paths and jumpsizes_adapted_vehs_paths
        for veh_idx in jumptimes_vehs_paths.keys():
            #RAW data
            jumptimes_veh = jumptimes_vehs_paths[veh_idx]
            jumpsizes_veh = jumpsizes_vehs_paths[veh_idx]
            #
            jump_or_nots_vehs = {}
            jumpsizes_adapted_vehs = {}
            #
            for path_idx in jumptimes_veh.keys():
                jumptimes = jumptimes_veh[path_idx]
                jumpsizes = jumpsizes_veh[path_idx]
                #
                #jump_or_nots and jumpsizes_adapted are two lists. 
                jump_or_nots = []
                jumpsizes_adapted = []
                #
                idx = 0
                for t in adapted_ts:
                    if t in jumptimes:
                        jump_or_nots.append(True)
                        jumpsizes_adapted.append(jumpsizes[idx])
                        idx = idx + 1
                    else:
                        jump_or_nots.append(False)
                        jumpsizes_adapted.append(0)
                    
                
                jump_or_nots_vehs[path_idx]  = jump_or_nots
                jumpsizes_adapted_vehs[path_idx] = jumpsizes_adapted
            #
            jump_or_nots_vehs_paths[veh_idx] = jump_or_nots_vehs
            jumpsizes_adapted_vehs_paths[veh_idx]  = jumpsizes_adapted_vehs
        
        return adapted_ts,jump_or_nots_vehs_paths,jumpsizes_adapted_vehs_paths
        

    
    @classmethod
    def AdaptEulerMultipleDims_ts4specificpath(slef, ts, jumptimes_vehs_paths_es, jumpsizes_vehs_paths_es):
        """
        adapt the times for jump and diffusion. 
        
        Difference:
        
            - self.AdaptEulerMultipleDims
            - self.AdaptEulerMultipleDims_ts4specificpath
            
        The latter one output adpated_ts for each path, while the former one use the adpated_ts for all paths. 
        
        
        Callback methods:
            
            #
            reload(irre)
            ts = np.linspace(0, 200, 400)
            #
            jumptimes_vehs_paths, jumpsizes_vehs_paths = self.JumpMomentsWithJumpSizes_multiplevehicle(T_horizon_sec = 4000, N_paths = 100)
            #
            adapted_ts_paths,jump_or_nots_vehs_paths,jumpsizes_adapted_vehs_paths = self.AdaptEulerMultipleDims_ts4specificpath(ts = ts, jumptimes_vehs_paths = jumptimes_vehs_paths, jumpsizes_vehs_paths = jumpsizes_vehs_paths)
        
        ---------------------------
        @input: ts
        
            a list. possiblly in equal distance. 
        
        @input: jumptimes_vehs_paths_es and jumpsizes_vehs_paths_es
            
            both are dict. 
            
            
            
            #
            jumptimes_vehs_paths = jumptimes_vehs_paths_es[dim_idx]
            jumpsizes_vehs_paths = jumpsizes_vehs_paths_es[dim_idx]
            
            dim_idx may mean longitudinal and lateral. 
            
            #
            both are dic of dict. 
            
            jumptimes = jumptimes_es[veh_idx][path_idx]
            jumpsizes = jumpsizes_es[veh_idx][path_idx]
            
            jumptimes and jumpsizes
                
                both are iteratible. 
                
                jumptimes = [t1, t2, t3....], each element is a float.
                jumpsizes = [js1, js2,....], each element is a float. 
                
            They are obtained via:
                jumptimes_vehs_paths, jumpsizes_vehs_paths = self.JumpMomentsWithJumpSizes_multiplevehicle(T_horizon_sec = 4000, N_paths = 100)
            

        
        @OUTPUT: adapted_ts_paths,jump_or_nots_vehs_paths_es,jumpsizes_adapted_vehs_paths_es
            
            
            adapted_ts = adapted_ts_paths[path_idx]
            adapted_ts is a list containing the moments.
            
            jump_or_nots_vehs_paths  = jump_or_nots_vehs_paths_es[dim_idx]
                
                jump_or_nots = jump_or_nots_vehs_paths[path_idx][veh_idx], is a list of True or False.
                    jump_or_nots is a list of True or False. 
                
                    If a element is True, then 
            
            jumpsizes_adapted_vehs_paths  = jumpsizes_adapted_vehs_paths_es[dim_idx]
                
                jumpsizes_adapted = jumpsizes_adapted_vehs_paths[path_idx][veh_idx],is either 0 or a jumptize. 
                    
                    jumpsizes_adapted is either 0 or a jumptize. 
        
        """
        #jump_or_nots_vehs_paths = jump_or_nots_vehs_paths_es[dim_idx]
        #   jump_or_nots_vehs_paths[veh_idx][path_idx]  = [True, False, ...]
        jump_or_nots_vehs_paths_es = []
        #jumpsizes_adapted_vehs_paths = jumpsizes_adapted_vehs_paths_es[idx]
        #   jumpsizes_adapted_vehs_paths[veh_idx][path_idx]  = [size1, size 2, ...]
        jumpsizes_adapted_vehs_paths_es = []
        #
        #Determine the Number of paths. 
        someveh_idx= list(jumpsizes_vehs_paths_es[0].keys())[0]
        paths_labels = jumpsizes_vehs_paths_es[0][someveh_idx].keys()
        #------------------------------------------adapted_ts_paths
        #adapted_ts_paths[path_idx] = [t0, t1, t2,....]
        adapted_ts_paths = {}
        for path in paths_labels:
            adapted_ts0 = list(ts)
            for jumptimes_vehs_paths in jumptimes_vehs_paths_es:
                for veh_idx in jumptimes_vehs_paths.keys():
                    adapted_ts0.extend(jumptimes_vehs_paths[veh_idx][path])
            adapted_ts_paths[path] = sorted(set(adapted_ts0))
        #------------------------------------------
        #
        #------------------------------------------jump_or_nots_vehs_paths and jumpsizes_adapted_vehs_paths
        #------------------------------------------
        #   jumptimes_vehs_paths[veh_idx][path_idx] is a list. 
        #   jumpsizes_vehs_paths[veh_idx][path_idx] is a list. 
        for jumptimes_vehs_paths,jumpsizes_vehs_paths in zip(jumptimes_vehs_paths_es, jumpsizes_vehs_paths_es):
            #-----------------Update, used to store the results.
            #jump_or_nots_vehs_paths[path_idx][veh_idx] = [True, False, ...]
            #jumpsizes_adapted_vehs_paths[path_idx][veh_idx] = [size1, size2, ...]
            jump_or_nots_vehs_paths = {}
            jumpsizes_adapted_vehs_paths = {}
            for veh_idx in jumptimes_vehs_paths.keys():
                #RAW data, jumptimes_veh is a dict. . jumpsizes_veh is a dict. 
                #   jumptimes_veh[path_idx] = [t1, t2,...]
                #   jumpsizes_veh[path_idx] = [s1, s2,...]
                jumptimes_veh = jumptimes_vehs_paths[veh_idx]
                jumpsizes_veh = jumpsizes_vehs_paths[veh_idx]
                #-----------------------Generate jump_or_nots_vehs and jumpsizes_adapted_vehs, both dicts. 
                #       jump_or_nots_vehs[path_idx] = a list. 
                #       jumpsizes_adapted_vehs[path_idx] = a list. 
                #   keys are path_idx
                jump_or_nots_vehs = {}
                jumpsizes_adapted_vehs = {}
                #
                for path_idx in jumptimes_veh.keys():
                    jumptimes = jumptimes_veh[path_idx]
                    jumpsizes = jumpsizes_veh[path_idx]
                    #
                    #jump_or_nots and jumpsizes_adapted are two lists. 
                    jump_or_nots = []
                    jumpsizes_adapted = []
                    #
                    idx = 0
                    for t in adapted_ts_paths[path_idx]:
                        if t in jumptimes:
                            jump_or_nots.append(True)
                            jumpsizes_adapted.append(jumpsizes[idx])
                            idx = idx + 1
                        else:
                            jump_or_nots.append(False)
                            jumpsizes_adapted.append(0)
                    #
                    jump_or_nots_vehs[path_idx]  = jump_or_nots
                    jumpsizes_adapted_vehs[path_idx] = jumpsizes_adapted
                #-----------------------DONE generate jump_or_nots_vehs and jumpsizes_adapted_vehs.
                #
                jump_or_nots_vehs_paths[veh_idx] = jump_or_nots_vehs
                jumpsizes_adapted_vehs_paths[veh_idx]  = jumpsizes_adapted_vehs
            #
            #Reverse the key, from [veh_idx][path_idx] to [path_idx][veh_idx].
            jump_or_nots_vehs_paths_key_reversed = {}
            for v in jump_or_nots_vehs_paths.keys():
                for p in jump_or_nots_vehs_paths[v].keys():
                    if p not in jump_or_nots_vehs_paths_key_reversed.keys():
                        jump_or_nots_vehs_paths_key_reversed[p] = {}
                    jump_or_nots_vehs_paths_key_reversed[p][v] = jump_or_nots_vehs_paths[v][p]
            jumpsizes_adapted_vehs_paths_key_reversed = {}
            for v in jumpsizes_adapted_vehs_paths.keys():
                for p in jumpsizes_adapted_vehs_paths[v].keys():
                    if p not in jumpsizes_adapted_vehs_paths_key_reversed.keys():
                        jumpsizes_adapted_vehs_paths_key_reversed[p] = {}
                    jumpsizes_adapted_vehs_paths_key_reversed[p][v] = jumpsizes_adapted_vehs_paths[v][p]
            #
            jump_or_nots_vehs_paths_es.append(jump_or_nots_vehs_paths_key_reversed)
            jumpsizes_adapted_vehs_paths_es.append(jumpsizes_adapted_vehs_paths_key_reversed)
            #
        #
        #adapted_ts_paths[path_idx] = [t0, t1, ...]
        #jump_or_nots_vehs_paths_es[dim_idx][veh_idx[path_idx] = [True, False,...]
        #jumpsizes_adapted_vehs_paths_es[dim_idx][veh_idx[path_idx] = [True, False,...]
        return adapted_ts_paths,jump_or_nots_vehs_paths_es,jumpsizes_adapted_vehs_paths_es

    
    @classmethod
    def AdaptEulerMultipleDims(slef, ts, jumptimes_vehs_paths_es, jumpsizes_vehs_paths_es):
        """
        adapt the times for jump and diffusion. 
        
        Callback methods:
            
            #
            reload(irre)
            ts = np.linspace(0, 200, 400)
            #
            jumptimes_vehs_paths, jumpsizes_vehs_paths = self.JumpMomentsWithJumpSizes_multiplevehicle(T_horizon_sec = 4000, N_paths = 100)
            #
            adapted_ts,jump_or_nots_vehs_paths,jumpsizes_adapted_vehs_paths = self.AdaptEuler(ts = ts, jumptimes_vehs_paths = jumptimes_vehs_paths, jumpsizes_vehs_paths = jumpsizes_vehs_paths)
        
        ---------------------------
        @input: ts
        
            a list. possiblly in equal distance. 
        
        @input: jumptimes_vehs_paths_es and jumpsizes_vehs_paths_es
            
            both are dict. 
            
            
            jumptimes_vehs_paths_es[veh_idx][path_idx] 
            
            #
            jumptimes_vehs_paths = jumptimes_vehs_paths_es[dim_idx]
            jumpsizes_vehs_paths = jumpsizes_vehs_paths_es[dim_idx]
            
            dim_idx may mean longitudinal and lateral. 
            
            #
            both are dic of dict. 
            
            jumptimes = jumptimes_es[veh_idx][path_idx]
            jumpsizes = jumpsizes_es[veh_idx][path_idx]
            
            jumptimes and jumpsizes
                
                both are iteratible. 
                
                jumptimes = [t1, t2, t3....], each element is a float.
                jumpsizes = [js1, js2,....], each element is a float. 
                
            They are obtained via:
                jumptimes_vehs_paths, jumpsizes_vehs_paths = self.JumpMomentsWithJumpSizes_multiplevehicle(T_horizon_sec = 4000, N_paths = 100)
            

        
        @OUTPUT: adapted_ts,jump_or_nots_vehs_paths_es,jumpsizes_adapted_vehs_paths_es
            
            three lists.
            
            adapted_ts is a list containing the moments.
            
            jump_or_nots_vehs_paths  = jump_or_nots_vehs_paths_es[idx]
                
                jump_or_nots = jump_or_nots_vehs_paths[veh_idx][path_idx], is a list of True or False.
                    jump_or_nots is a list of True or False. 
                
                    If a element is True, then 
            
            jumpsizes_adapted_vehs_paths  = jumpsizes_adapted_vehs_paths_es[idx]
                
                jumpsizes_adapted = jumpsizes_adapted_vehs_paths[veh_idx][path_idx],is either 0 or a jumptize. 
                    
                    jumpsizes_adapted is either 0 or a jumptize. 
        
        """
        #jump_or_nots_vehs_paths = jump_or_nots_vehs_paths_es[idx]
        #   jump_or_nots_vehs_paths[veh_idx][path_idx]  = [True, False, ...]
        jump_or_nots_vehs_paths_es = []
        #jumpsizes_adapted_vehs_paths = jumpsizes_adapted_vehs_paths_es[idx]
        #   jumpsizes_adapted_vehs_paths[veh_idx][path_idx]  = [size1, size 2, ...]
        jumpsizes_adapted_vehs_paths_es = []
        
        #------------------------------------------adapted_ts
        adapted_ts0 = list(ts)
        for jumptimes_vehs_paths in jumptimes_vehs_paths_es:
            for veh_idx in jumptimes_vehs_paths.keys():
                #
                jumptimes_veh = jumptimes_vehs_paths[veh_idx]
                for path_idx in jumptimes_veh.keys():
                    #
                    jumptimes = jumptimes_veh[path_idx]
                    #
                    adapted_ts0.extend(jumptimes)
        #
        adapted_ts = sorted(set(adapted_ts0))
        #
        #------------------------------------------jump_or_nots_vehs_paths and jumpsizes_adapted_vehs_paths
        for jumptimes_vehs_paths,jumpsizes_vehs_paths in zip(jumptimes_vehs_paths_es, jumpsizes_vehs_paths_es):
            #-----------------Update 
            jump_or_nots_vehs_paths = {}
            jumpsizes_adapted_vehs_paths = {}
            for veh_idx in jumptimes_vehs_paths.keys():
                #RAW data, jumptimes_veh is a dict. . jumpsizes_veh is a dict. 
                #   jumptimes_veh[path_idx] = [t1, t2,...]
                #   jumpsizes_veh[path_idx] = [s1, s2,...]
                jumptimes_veh = jumptimes_vehs_paths[veh_idx]
                jumpsizes_veh = jumpsizes_vehs_paths[veh_idx]
                #-----------------------Generate jump_or_nots_vehs and jumpsizes_adapted_vehs, both dicts. 
                #       jump_or_nots_vehs[path_idx] = a list. 
                #       jumpsizes_adapted_vehs[path_idx] = a list. 
                #   keys are path_idx
                jump_or_nots_vehs = {}
                jumpsizes_adapted_vehs = {}
                #
                for path_idx in jumptimes_veh.keys():
                    jumptimes = jumptimes_veh[path_idx]
                    jumpsizes = jumpsizes_veh[path_idx]
                    #
                    #jump_or_nots and jumpsizes_adapted are two lists. 
                    jump_or_nots = []
                    jumpsizes_adapted = []
                    #
                    idx = 0
                    for t in adapted_ts:
                        if t in jumptimes:
                            jump_or_nots.append(True)
                            jumpsizes_adapted.append(jumpsizes[idx])
                            idx = idx + 1
                        else:
                            jump_or_nots.append(False)
                            jumpsizes_adapted.append(0)
                    #
                    jump_or_nots_vehs[path_idx]  = jump_or_nots
                    jumpsizes_adapted_vehs[path_idx] = jumpsizes_adapted
                #-----------------------DONE generate jump_or_nots_vehs and jumpsizes_adapted_vehs.
                #
                jump_or_nots_vehs_paths[veh_idx] = jump_or_nots_vehs
                jumpsizes_adapted_vehs_paths[veh_idx]  = jumpsizes_adapted_vehs
            #
            jump_or_nots_vehs_paths_es.append(jump_or_nots_vehs_paths)
            jumpsizes_adapted_vehs_paths_es.append(jumpsizes_adapted_vehs_paths)
            #
        return adapted_ts,jump_or_nots_vehs_paths_es,jumpsizes_adapted_vehs_paths_es




    @classmethod
    def AdaptEuler_BKP2(slef, ts, jumptimes_lists, jumpsizes_lists):
        """
        adapt the times for jump and diffusion. 
        
        Callback methods:
        
            adapted_ts,jump_or_nots_lists,jumpsizes_adapted_lists = self.AdaptEuler(ts = ts, jumptimes_lists = jumptimes_lists, jumpsizes_lists = jumpsizes_lists)
            
            jump_or_nots_lists and jumpsizes_adapted_lists are the same lengnth as jumpsizes_lists. 
        
        ---------------------------
        @input: ts
        
            a list. possiblly in equal distance. 
        
        @input: jumptimes_lists, jumpsizes_lists
        
            jumptimes = jumptimes_lists[idx]
            
                jumptimes is a list containing the jump moments generated by self.JumpMomentsWithJumpSizes()
                
            jumpsizes = jumpsizes_lists[idx]
                
                jump size is a list containing the jump sizes generated by self.JumpMomentsWithJumpSizes()
                
                
            They are obtained via:
            
                
        
        @OUTPUT: adapted_ts,jump_or_nots_lists,jumpsizes_adapted_lists
            
            three lists.
            
            adapted_ts is a list containing the moments.
            
            jump_or_nots = jump_or_nots_lists[veh_idx], is a list of True or False.
                jump_or_nots is a list of True or False. 
            
                If a element is True, then 
            
            jumpsizes_adapted = jumpsizes_adapted_lists[idx],is either 0 or a jumptize. 
                
                jumpsizes_adapted is either 0 or a jumptize. 
        
        """
        jump_or_nots_lists = []
        jumpsizes_adapted_lists = []
        
        #------------------------------------------adapted_ts
        adapted_ts0 = list(ts)
        for jumptimes in jumptimes_lists:
            adapted_ts0.extend(jumptimes)
        adapted_ts = sorted(set(adapted_ts0))
        
        #------------------------------------------
        for jumptimes,jumpsizes in zip(jumptimes_lists,jumpsizes_lists):
            jump_or_nots = []
            jumpsizes_adapted = []
            idx = 0
            for t in adapted_ts:
                if t in jumptimes:
                    jump_or_nots.append(True)
                    #print(idx, len(jumpsizes))
                    jumpsizes_adapted.append(jumpsizes[idx])
                    idx = idx + 1
                else:
                    jump_or_nots.append(False)
                    jumpsizes_adapted.append(0)
            #
            jump_or_nots_lists.append(jump_or_nots)
            jumpsizes_adapted_lists.append(jumpsizes_adapted)
            
        #
        return adapted_ts,jump_or_nots_lists,jumpsizes_adapted_lists
        
    @classmethod
    def JumpMomentsWithJumpSizes_singlevehicle(self, jumpdensity, jumpdis_type = 'Normal', args_jumpsize_dis = {'mu':.0, 'sigma': .1}, T_horizon_sec = 4000):
        """
        Generate the jump times and the jump sizes. 
        
            jumptimes,jumpsizes = self.JumpMomentsWithJumpSizes(T_horizon_sec = 4000)
        
        -------------------------------------
        @input: jumpdensity
            
            average time between neighboring jumps. 
            
            the parameter of the poisson distribution, 
            
            
            np.random.exponential(scale = jumpdensity,size=1000) = jumpdensity
             
        @input: jumpdis_type and args_jumpsize_dis
        
            jumpdis_type 
            
                is the type of the jump size distribtion.
                
                'Normal' is normal distribution. 
                
            args_jumpsize_dis
            
                is the parameters of the jump size distribution. 
        
        @input: T_horizon_sec
        
            the time horizon. unit is second. 
        
        @OUTPUT: jumptimes,jumpsizes
        
            both are iteratible. 
            
            jumptimes = [t1, t2, t3....], each element is a float.
            jumpsizes = [js1, js2,....], each element is a float. 
            
        """
        jumptimes = []
        jumpsizes = []
        
        while len(jumptimes)==0 or min(jumptimes)<T_horizon_sec:
            #
            t = np.random.exponential(scale = jumpdensity)
            jumptimes.append(t)
            #
            #
            if jumpdis_type=='normal':
                jumpsize = np.random.normal(loc = args_jumpsize_dis['mu'], scale = args_jumpsize_dis['sigma'])
                jumpsizes.append(jumpsize)
                
        return jumptimes,jumpsizes
    
    @classmethod
    def JumpMomentsWithJumpSizes_multiplevehicles(self, jumpdensities = [8], jumpdis_types = ['Normal'], args_jumpsize_diss = [{'mu':.0, 'sigma': .1}], T_horizon_sec = 4000, N_paths = 100):
        """
        Generate the jump times and the jump sizes. 
        
        Callback method:
        
            #
            #jumptimes_es_es[veh_idx][path_idx] = [t0, t1...]
            #jumpsizes_es_es[veh_idx][path_idx] = [size0, size1...]
            jumptimes_es_es,jumpsizes_es_es = self.JumpMomentsWithJumpSizes_multiplevehicle(T_horizon_sec = 4000, N_paths = 100)
        
        -------------------------------------
        @input: jumpdensity
        
            jumpdensity[veh_idx] is a float. 
            
            a list containing average time between neighboring jumps. 
            
            The length is the number of vehicles. 
            
            the parameter of the poisson distribution, 
            
            
            np.random.exponential(scale = jumpdensity,size=1000) = jumpdensity
             
        @input: jumpdis_type and args_jumpsize_dis
        
            jumpdis_type 
            
                is the type of the jump size distribtion.
                
                'Normal' is normal distribution. 
                
            args_jumpsize_dis
            
                is the parameters of the jump size distribution. 
        
        @input: T_horizon_sec
        
            the time horizon. unit is second. 
        
        @OUTPUT: jumptimes_es_es,jumpsizes_es_es
        
            a dict of dict. 
            
            jumptimes = jumptimes_es[veh_idx][path_idx]
            jumpsizes = jumpsizes_es[veh_idx][path_idx]
            
            veh_idx is counted from 0  to len(jumpdensities).
            
            jumptimes and jumpsizes
                
                both are iteratible. 
                
                jumptimes = [t1, t2, t3....], each element is a float.
                jumpsizes = [js1, js2,....], each element is a float. 
            
        """
        jumptimes_es_es = {}
        jumpsizes_es_es = {}
        
        #
        #for each vehicle
        for veh_idx,(jumpdensity,jumpdis_type,args_jumpsize_dis) in enumerate(zip(jumpdensities, jumpdis_types,args_jumpsize_diss)):
            jumptimes_es = {}
            jumpsizes_es = {}
            
            for path_idx in range(N_paths):
                #jumptimes and jumpsizes are both lists of floats. 
                jumptimes = []
                jumpsizes = []
                #
                while len(jumptimes)==0 or max(jumptimes)<T_horizon_sec:
                    #get jumptime
                    if len(jumptimes)==0:
                        previous_time = 0
                    else:
                        previous_time = jumptimes[-1]
                    #
                    t = np.random.exponential(scale = jumpdensity)
                    if previous_time + t>T_horizon_sec:break
                    #
                    jumptimes.append( previous_time + t)
                    #
                    #get jumpsize
                    if jumpdis_type=='Normal':
                        jumpsize = np.random.normal(loc = args_jumpsize_dis['mu'], scale = args_jumpsize_dis['sigma'])
                        jumpsizes.append(jumpsize)
                    
                #
                jumptimes_es[path_idx] = jumptimes
                jumpsizes_es[path_idx] = jumpsizes
            #
            jumptimes_es_es[veh_idx] = jumptimes_es
            jumpsizes_es_es[veh_idx] = jumpsizes_es
            #
        return jumptimes_es_es,jumpsizes_es_es
        

class SingleLaneSim():
    """
    Signal lane simulation of the jump diffuction. 
    
    """
    
    
    
    @classmethod
    def TrimUpdate_single_dim_with_jump(self, old_state, deltat_sec, F, diffusion_L, brownian, idm_paras = idm_paras, jumpsize = .0, jerk_constraint = False, old_old_state = False, old_delta_sec = False):
        """
        single_dim means just longitudinal . 
        
        The update of the state is calculated as:
        
            new_state = old_state + F*deltat_sec + np.matmul(L, brownian)
            
        Or equilivalently:
        
            new_state = old_state + (F + np.matmul(L, brownian/deltat_sec))*deltat_sec
        
        The above is for the sysmtem dynamics dS = F(S)dt + L(S) dW. 
        
        NOTE that the old_state is defined as:
        
            - x,vx,y,vy,zlon,zlat
        
        However, the acceleration may exceed the maximum acceleration. 
        
        -----------------------------------
        @input: diffusion_L
        
            diffusion_L is a 1d array, the same size as F. 
            
                da = F + diffusion_L*dW
            
        @input: idm_acce
            
            float, which represent the acceleration calculated from IDM model. 
            
        @input: jumpsize
        
            the jump size of the acceleration, i.e. 
            
        @input: deltat_sec
        
            a float. 
        
        @OUTPUT: new_state,feasible_increment
        
            both are 1d array. 
        
        """
        increment = F + diffusion_L*brownian/deltat_sec + np.array([.0, jumpsize, .0, .0, .0, .0])/deltat_sec
        """
        if jump_or_not:
            #increment shape is (6,)
            increment = F + np.matmul(diffusion_L, brownian/deltat_sec) + np.array([.0, jumpsize, .0, .0, .0, .0])/deltat_sec
        else:
            #increment shape is (6,)
            increment = F + diffusion_L*brownian/deltat_sec#  np.matmul(L, brownian/deltat_sec)
        """
        
        #the speed. must be positive. 
        increment[0] = max(0, increment[0])
        
        #the acceelration, must be fall in reasonable interval 
        increment[1] = min(idm_paras['a_MAX'], max(-idm_paras['a_MAX'], increment[1]))
        
        #the jerk
        if jerk_constraint:
            v0 = old_old_state[1]
            v1 = old_state[1]
            deltat0 = old_delta_sec
            deltat1 = deltat_sec
            #
            v_minn,v_maxx = TwoDimMicroModel.Speed_upperlower_given_jerk_upperlower(v0 = v0, v1 = v1, deltat0 = deltat0, deltat1 = deltat1, )
            #
            increment[1] = min((v_maxx-v1)/deltat_sec, max((v_minn-v1)/deltat_sec, increment[1]))
            pass
            #old_old_state = False, old_delta_sec = False
        
        feasible_increment = increment*deltat_sec
        
        return old_state + increment*deltat_sec,feasible_increment
    
    @classmethod
    def TrimUpdate_single_dim(self, old_state, deltat_sec, F, diffusion_L, brownian, idm_paras = idm_paras, jumpsize = .0, jump_or_not = False):
        """
        single_dim means just longitudinal . 
        
        The update of the state is calculated as:
        
            new_state = old_state + F*deltat_sec + np.matmul(L, brownian)
            
        Or equilivalently:
        
            new_state = old_state + (F + np.matmul(L, brownian/deltat_sec))*deltat_sec
        
        The above is for the sysmtem dynamics dS = F(S)dt + L(S) dW. 
        
        NOTE that the old_state is defined as:
        
            - x,vx,y,vy,zlon,zlat
        
        However, the acceleration may exceed the maximum acceleration. 
        
        -----------------------------------
        @input: diffusion_L
        
            diffusion_L is a 1d array. the same as F. 
            
                da = F + diffusion_L*dW
            
        @input: idm_acce
            
            float, which represent the acceleration calculated from IDM model. 
            
        @input: jumpsize
        
            the jump size of the acceleration, i.e. 
            
        @input: deltat_sec
        
            a float. 
        
        @OUTPUT: new_state,feasible_increment
        
            both are 1d array. 
        
        """
        if jump_or_not:
            #increment shape is (6,)
            increment = F + np.matmul(diffusion_L, brownian/deltat_sec) + np.array([.0, jumpsize, .0, .0, .0, .0])/deltat_sec
            
            
        else:
            #increment shape is (6,)
            increment = F + diffusion_L*brownian/deltat_sec#  np.matmul(L, brownian/deltat_sec)
        
        #the speed. must be positive. 
        increment[0] = max(0, increment[0])
        
        #the acceelration, must be fall in reasonable interval 
        increment[1] = min(idm_paras['a_MAX'], max(-idm_paras['a_MAX'], increment[1]))
        
        feasible_increment = increment*deltat_sec
        
        return old_state + increment*deltat_sec,feasible_increment

    @classmethod
    def TrimUpdate(self, old_state, deltat_sec, F, L, brownian, idm_paras = idm_paras, jumpsize = .0, jump_or_not = True, jerk_constraint = False, old_old_state = False, old_delta_sec = False):
        """
        The update of the state is calculated as:
        
            new_state = old_state + F*deltat_sec + np.matmul(L, brownian)
            
        Or equilivalently:
        
            new_state = old_state + (F + np.matmul(L, brownian/deltat_sec))*deltat_sec
        
        The above is for the sysmtem dynamics dS = F(S)dt + L(S) dW. 
        
        NOTE that the old_state is defined as:
        
            - x,vx,y,vy,zlon,zlat
        
        However, the acceleration may exceed the maximum acceleration. 
        
        -----------------------------------
        @input: jerk_constraint
        
            a bool. Used to indicate that whether the max and min jerk constraint is applied. 
            
            
                minn,maxx = TwoDimMicroModel.Speed_upperlower_given_jerk_upperlower(v0, v1, deltat0, delta1, )
        
        @input: jumpsize
        
            the jump size of the acceleration, i.e. 
            
        @input: deltat_sec
        
            a float. 
        
        @input: F, L
        
            both are array. 
            
            F.shape is (6,)
            
            L.shape is (6, 2)
            
            brownian shape is (2,)
            
            
        
        """
        if jump_or_not:
            #increment shape is (6,)
            increment = F + np.matmul(L, brownian/deltat_sec) + np.array([.0, jumpsize, .0, .0, .0, .0])/deltat_sec
        else:
            #increment shape is (6,)
            increment = F + np.matmul(L, brownian/deltat_sec)
        
        #the speed. must be positive. 
        increment[0] = max(0, increment[0])
        
        
        #the acceelration, must be fall in reasonable interval 
        increment[1] = min(idm_paras['a_MAX'], max(-idm_paras['a_MAX'], increment[1]))
        
        #the jerk
        if jerk_constraint:
            v0 = old_old_state[1]
            v1 = old_state[1]
            #
            deltat0 = old_delta_sec
            deltat1 = deltat_sec
            #
            v_minn,v_maxx = TwoDimMicroModel.Speed_upperlower_given_jerk_upperlower(v0 = v0, v1 = v1, deltat0 = deltat0, deltat1 = deltat1, )
            #
            increment[1] = min((v_maxx-v1)/deltat_sec, max((v_minn-v1)/deltat_sec, increment[1]))
            pass
            #old_old_state = False, old_delta_sec = False
        
        feasible_increment = increment*deltat_sec
        
        return old_state + increment*deltat_sec,feasible_increment

    
    
    @classmethod
    def TrimUpdate_two_dim_jump(self, old_state, deltat_sec, F, L, brownian, idm_paras = idm_paras, jumpsize_lon = .0, jumpsize_lat = .0, jump_or_not_lon = True, jump_or_not_lat = True, old_old_state = False, old_delta_sec = False, jerk_constraint_lon = False, jerk_constraint_lat = False):
        """
        Difference:
        
            self.TrimUpdate
            self.TrimUpdate_twodim_jump
            
        The latter have four args: jumpsize_lon, jump_lat, jump_or_not_lon and jump_or_not_lat
        
        
        The update of the state is calculated as:
        
            new_state = old_state + F*deltat_sec + np.matmul(L, brownian)
            
        Or equilivalently:
        
            new_state = old_state + (F + np.matmul(L, brownian/deltat_sec))*deltat_sec
        
        The above is for the sysmtem dynamics dS = F(S)dt + L(S) dW. 
        
        NOTE that the old_state is defined as:
        
            - x,vx,y,vy,zlon,zlat
        
        However, the acceleration may exceed the maximum acceleration. 
        
        -----------------------------------
        @input: jerk_constraint
        
            a bool. Used to indicate that whether the max and min jerk constraint is applied. 
            
            
                minn,maxx = TwoDimMicroModel.Speed_upperlower_given_jerk_upperlower(v0, v1, deltat0, delta1, )
        
        @input: jumpsize
        
            the jump size of the acceleration, i.e. 
            
        @input: deltat_sec
        
            a float. 
        
        @input: F, L
        
            both are array. 
            
            F.shape is (6,)
            
            L.shape is (6, 2)
            
            brownian shape is (2,)
            
            
        
        """
        if jump_or_not_lon and jump_or_not_lat:
            #increment shape is (6,)
            #print(L.shape, )
            increment = F + np.matmul(L, brownian/deltat_sec) + np.array([.0, jumpsize_lon, .0, jumpsize_lat, .0, .0])/deltat_sec
        elif (not jump_or_not_lon) and jump_or_not_lat:
            #increment shape is (6,)
            increment = F + np.matmul(L, brownian/deltat_sec) + np.array([.0, .0, .0, jumpsize_lat, .0, .0])/deltat_sec
        elif jump_or_not_lon and (not jump_or_not_lat):
            #increment shape is (6,)
            increment = F + np.matmul(L, brownian/deltat_sec) + np.array([.0, jumpsize_lon, .0, .0, .0, .0])/deltat_sec
        else:
            #increment shape is (6,)
            increment = F + np.matmul(L, brownian/deltat_sec)
        
        #the speed. must be positive. 
        increment[0] = max(0, increment[0])
        
        #
        #the acceelration, must be fall in reasonable interval 
        increment[1] = min(idm_paras['a_MAX'], max(-idm_paras['a_MAX'], increment[1]))
        
        #the jerk
        if jerk_constraint_lon:
            #==============================================lon
            v0 = old_old_state[1];v1 = old_state[1]
            deltat0 = old_delta_sec;deltat1 = deltat_sec
            #
            v_minn,v_maxx = TwoDimMicroModel.Speed_upperlower_given_jerk_upperlower(v0 = v0, v1 = v1, deltat0 = deltat0, deltat1 = deltat1, )
            #
            increment[1] = min((v_maxx-v1)/deltat_sec, max((v_minn-v1)/deltat_sec, increment[1]))
            
        if jerk_constraint_lat:
            #==============================================lat
            v0 = old_old_state[3];v1 = old_state[3]
            deltat0 = old_delta_sec;deltat1 = deltat_sec
            #
            v_minn,v_maxx = TwoDimMicroModel.Speed_upperlower_given_jerk_upperlower_lat(v0 = v0, v1 = v1, deltat0 = deltat0, deltat1 = deltat1, )
            #
            increment[3] = min((v_maxx-v1)/deltat_sec, max((v_minn-v1)/deltat_sec, increment[3]))
            #
            #old_old_state = False, old_delta_sec = False
        
        feasible_increment = increment*deltat_sec
        
        return old_state + increment*deltat_sec,feasible_increment


    @classmethod
    def BoundaryOfSpeedJerkPlane(self, simpaths):
        """
        @input: simpaths
        
            #sim_paths[veh_idx][path_idx] = pd, columns are moments, which is the same as input arg ts.
            
        """
        #get (vs, jerks) pair, where vs are sorted. 
        vs = []
        jerks = []
        for vid in simpaths.keys():
            #
            for pathidx in simpaths[vid].keys():
                
                
                
                pass
            
            pass
        
        
        
        
        pass

    def Get_QK(self, snapshots, Deltat_FD_sec = 30):
        """
        calculate the Qand K
        
            = 
        
        @input: snapshots
        
            snapshots[moment].kesys() are:
            
                self.snapshots[T]['vehs_at_lanes'] = copy.deepcopy(self.vehs_at_lanes)
                self.snapshots[T][vid] = {'S':copy.deepcopy(self.vehs_dict[vid]), \
                    #'feasibleincrement':copy.deepcopy(feasible_increment), \
                    'L':copy.deepcopy(L_dicts[vid]), \
                    'F':copy.deepcopy(F_dicts[vid]), \
                    'vehs_info_within_area_left':copy.deepcopy(vehs_info_within_area_left_dict[vid]), \
                    'vehs_info_within_area_right':copy.deepcopy(vehs_info_within_area_right_dict[vid]), \
                    'potentials':copy.deepcopy(potentials_dict[vid]), }
        
        @input: Deltat_FD_sec
        
            the time step that calculate the FD parameters. 
        
        @OUTPUT: Q and K
        
            both are dicts. 
            
            Q[laneid] = [q1, q2, q3, q4...]
            K[laneid] = [k1, k2, k3, k4...]
            
            Q unit is veh/h
            K unit is veh/km
        
        
        
        
        -----------------------------
        Q = dA/|A|, d is the distance travelled by all vehicles.
        K = t(A)/|A|, t is the total time travelled. 
        """
        #Q[laneid] = [q1, q2, q3, q4...]
        #K[laneid] = [k1, k2, k3, k4...]
        t = sorted(snapshots.keys())[0]
        Q = {lane_id:[] for lane_id in snapshots[t]['vehs_at_lanes'].keys()}
        K = {lane_id:[] for lane_id in snapshots[t]['vehs_at_lanes'].keys()}
        
        #unit is km.h
        area = (self.length/1000.0)*(Deltat_FD_sec/3600.0)
        #
        Ts = sorted(snapshots.keys())
        #
        start = 0.0
        end = start + Deltat_FD_sec
        while end<=max(Ts):
            #unit is sec and meter.
            totaltimetravelled_sec = 0
            totaldistancetravelled_meter = 0
            #
            interval_ts = sorted(Ts[Ts>start and Ts<=end])
            for t0,t1 in zip(interval_ts[:-1], interval_ts[1:]):
                #
                for lane_id in snapshots[t]['vehs_at_lanes'].keys():
                    #
                    for vid in snapshots[t]['vehs_at_lanes'][lane_id]:
                        #time travelled
                        totaltimetravelled_sec = totaltimetravelled_sec + t1 - t0
                        #distancetravelled
                        if snapshots[t1][vid]['S'][0]<=snapshots[t0][vid]['S'][0]:
                            distance_travelled_vid = snapshots[t1][vid]['S'][0] + self.length - snapshots[t0][vid]['S'][0]
                        else:
                            distance_travelled_vid = snapshots[t1][vid]['S'][0]  - snapshots[t0][vid]['S'][0]
                        #
                        totaldistancetravelled_meter = totaldistancetravelled_meter  + distance_travelled_vid
                #
                Q_interval = totaldistancetravelled_meter/1000.0/area
                K_interval = totaldistancetravelled_meter/3600.0/area
                #
                Q[lane_id].append(Q_interval)
                K[lane_id].append(K_interval)
            start = end
        
        return Q,K
        


    def get_lanes_densities(self, ):
        """
        Get the density of all lanes.
        
        -------------------------------
        @output: lanesdensities
        
            lanesdensities[lane_id] = float.
        
        """
        
        return len(self.vehs_dict)/(self.length/1000.0)
    

    def __init__(self, lw = 3.5, length = 500):
        """
        
        ---------------------------------------------------------------
        @input: lws
        
            a lists containing the lane withs of each lane. 
            
            
            
        """
        #======================================self.road_bounds
        self.road_bounds = (0, lw)
        #lw_boundary_lanes
        #self.lw_boundary_lanes = (self.lws[0], self.lws[-1])
        
        
        
        #========================================self.lanes_ids, self.lws, self.length
        #   lane id
        self.lw = lw
        self.length = length
        

        #======================================self.road_bounds

        
        #========================================Dynmaic properties. 
        #   ----------------self.vehs_at_lanes, self.vehs_dict,self.vehs_target_lane
        #self.vehs_at_lanes= [vid1, vid2,....]. NOTE THAT THEY ARE SPORTED FROM 1st vehicle to last vehicle. 
        self.vehs_at_lanes  =  []
        #self.vehs_dict[vehicle_id] = [x, vx, y, vy, zlon, zlat]
        self.vehs_dict = {}
        #   self.snapshots[moment] is a dict. 
        #   self.snapshots[moment][vid] is a dict. 
        #   self.snapshots[moment][vid] keys include 'leader','neighbores','potentials.'
        self.snapshots = {}
        #-------------------------------


    @classmethod
    def plot_opensim_leaderstates_followerpaths_im_heatmap4jerk(self, followerssim_paths, axs = False, leaderstates = False, figsize = (5,3), alpha = .4, vmax = 200/3.6, quantiles = [.5], loc_idx = 0, v_idx = 1,  sigma4heatplot = 10, bins4heatplot=(200, 200) ):
        """
        
        plot the simulation on open road. 
        
        -------------------------------
        
        @input: leaderstates
            
            leaderstates[moment] is a 1d array. [x, vx, y, vy, zlon, zlat]
            
            leaderstates = irre.SingleLaneSim.LeaderVehicleState_constancespeed(ts = ts, v_ms = v/3.6)
        
        
        @input: followersim_paths
            
            followerssim_paths[veh_idx][v][path_idx] = pd, columns are moments, which is the same as input arg ts.
            
            followersim_paths = irre.SingleLaneSim.sim_openroad_pure_HDVs(ts = ts, init_states = init_states, leader_states = leaderstates,  \
                    N_paths = 200, tao_delay = .8, acc_model = 'IDM', diffusion_typee = 'speed_dependent_jerk')
            
        """
        
        if isinstance(axs, bool):
            fig,axs = plt.subplots(figsize = figsize, ncols = 2, nrows = 2)
            #ax.set_xlabel('Time (sec)');ax.set_ylabel('x (m)'); 
            #ax.grid();
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        #
        #####################x
        ax = axs[0,0]
        ax.set_xlabel('Time (sec)');ax.set_ylabel('x (m)'); 
        if not isinstance(leaderstates, bool):
            xs_leader = np.array([leaderstates[t][loc_idx] for t in sorted(leaderstates.keys())])
            ax.plot(sorted(leaderstates.keys()), xs_leader, '.-k', alpha = alpha, )
        #
        #
        for veh_idx in followerssim_paths.keys():
            for path_idx in followerssim_paths[veh_idx].keys():
                #
                xs = followerssim_paths[veh_idx][path_idx].iloc[loc_idx, :]
                #
                ax.plot(xs.index, xs.values, alpha = alpha)
        ax.grid('on')
        #####################v
        ax = axs[0, 1]
        ax.set_xlabel('Time (sec)');ax.set_ylabel('v (m/s)'); 
        if not isinstance(leaderstates, bool):
            vs_leader = np.array([leaderstates[t][v_idx] for t in sorted(leaderstates.keys())])
            ax.plot(sorted(leaderstates.keys()), vs_leader, '.-k', alpha = alpha, )
        #
        #
        for veh_idx in followerssim_paths.keys():
            for path_idx in followerssim_paths[veh_idx].keys():
                #
                vs = followerssim_paths[veh_idx][path_idx].iloc[v_idx, :]
                #
                ax.plot(vs.index, vs.values, alpha = alpha)
        ax.grid('on')
        #####################a
        ax = axs[1, 0]
        ax.set_xlabel('Time (sec)');ax.set_ylabel('$accelerations (m/s^2)$'); 
        #
        for veh_idx in followerssim_paths.keys():
            for path_idx in followerssim_paths[veh_idx].keys():
                #
                vs = followerssim_paths[veh_idx][path_idx].iloc[v_idx, :]
                ts = vs.index.values
                #
                ax.plot(ts[1:],np.diff(vs.values)/np.diff(ts), alpha = alpha)
        ax.grid('on')
        #####################jerk
        ax = axs[1, 1]
        ax.set_xlabel('Time (sec)');ax.set_ylabel('$jerk (m/s^{3}})$'); 
        #
        JERKS = []
        TS = []
        for veh_idx in followerssim_paths.keys():
            for path_idx in followerssim_paths[veh_idx].keys():
                #
                vs = followerssim_paths[veh_idx][path_idx].iloc[v_idx, :]
                ts = vs.index.values
                #
                ass = np.diff(vs.values)/np.diff(ts)
                JERKS.extend(list(np.diff(ass)/np.diff(ts[1:])))
                TS.extend(list(ts[2:]))
                #
                #ax.plot(ts[2:],np.diff(ass)/np.diff(ts[1:]), '.',)
        #
        img, extent = myplot(x = TS, y = JERKS, sigma = sigma4heatplot, bins=bins4heatplot)
        ax.imshow(img, extent = extent, origin='lower', cmap=cm.jet,  aspect='auto', interpolation='nearest')
        ax.grid('on')
        #
        #
        return axs
        
    @classmethod
    def plot_opensim_leaderstates_followerpaths_im(self, followerssim_paths, axs = False, leaderstates = False, figsize = (5,3), alpha = .4, vmax = 200/3.6, quantiles = [.5], loc_idx = 0, v_idx = 1, n_acceleration_plotted = False):
        """
        
        plot the simulation on open road. 
        
        -------------------------------
        
        @input: leaderstates
            
            leaderstates[moment] is a 1d array. [x, vx, y, vy, zlon, zlat]
            
            leaderstates = irre.SingleLaneSim.LeaderVehicleState_constancespeed(ts = ts, v_ms = v/3.6)
        
        
        @input: followersim_paths
            
            followerssim_paths[veh_idx][v][path_idx] = pd, columns are moments, which is the same as input arg ts.
            
            followersim_paths = irre.SingleLaneSim.sim_openroad_pure_HDVs(ts = ts, init_states = init_states, leader_states = leaderstates,  \
                    N_paths = 200, tao_delay = .8, acc_model = 'IDM', diffusion_typee = 'speed_dependent_jerk')
            
        """
        
        if isinstance(axs, bool):
            fig,axs = plt.subplots(figsize = figsize, ncols = 2, nrows = 2)
            #ax.set_xlabel('Time (sec)');ax.set_ylabel('x (m)'); 
            #ax.grid();
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        #
        #####################x
        ax = axs[0,0]
        ax.set_xlabel('Time (sec)');ax.set_ylabel('x (m)'); 
        if not isinstance(leaderstates, bool):
            xs_leader = np.array([leaderstates[t][loc_idx] for t in sorted(leaderstates.keys())])
            ax.plot(sorted(leaderstates.keys()), xs_leader, '.-k', alpha = alpha, )
        #
        #
        for veh_idx in followerssim_paths.keys():
            for path_idx in followerssim_paths[veh_idx].keys():
                #
                xs = followerssim_paths[veh_idx][path_idx].iloc[loc_idx, :]
                #
                ax.plot(xs.index, xs.values, alpha = alpha)
        ax.grid('on')
        #####################v
        ax = axs[0, 1]
        ax.set_xlabel('Time (sec)');ax.set_ylabel('v (m/s)'); 
        if not isinstance(leaderstates, bool):
            vs_leader = np.array([leaderstates[t][v_idx] for t in sorted(leaderstates.keys())])
            ax.plot(sorted(leaderstates.keys()), vs_leader, '.-k', alpha = alpha, )
        #
        #
        for veh_idx in followerssim_paths.keys():
            for path_idx in followerssim_paths[veh_idx].keys():
                #
                vs = followerssim_paths[veh_idx][path_idx].iloc[v_idx, :]
                #
                ax.plot(vs.index, vs.values, alpha = alpha)
        ax.grid('on')
        #####################acceleration
        ax = axs[1, 0]
        ax.set_xlabel('Time (sec)');ax.set_ylabel('$accelerations (m/s^2)$'); 
        #
        if isinstance(n_acceleration_plotted, bool):
            for veh_idx in followerssim_paths.keys():
                for path_idx in followerssim_paths[veh_idx].keys():
                    #
                    vs = followerssim_paths[veh_idx][path_idx].iloc[v_idx, :]
                    ts = vs.index.values
                    #
                    ax.plot(ts[1:],np.diff(vs.values)/np.diff(ts), alpha = alpha)
        else:
            
            selected = np.random.choice(list(followerssim_paths[veh_idx].keys()), min(n_acceleration_plotted, len(followerssim_paths[veh_idx].keys())))
            for veh_idx in followerssim_paths.keys():
                for path_idx in selected:
                    #
                    vs = followerssim_paths[veh_idx][path_idx].iloc[v_idx, :]
                    ts = vs.index.values
                    #
                    ax.plot(ts[1:],np.diff(vs.values)/np.diff(ts), alpha = alpha)
            
            
            pass

        ax.grid('on')
        #####################jerk
        ax = axs[1, 1]
        ax.set_xlabel('Speed (m/s)');ax.set_ylabel('$jerk (m/s^{3}}))$'); 
        #
        for veh_idx in followerssim_paths.keys():
            for path_idx in followerssim_paths[veh_idx].keys():
                #
                vs = followerssim_paths[veh_idx][path_idx].iloc[v_idx, :]
                ts = vs.index.values
                #
                ass = np.diff(vs.values)/np.diff(ts)
                
                #
                #ax.plot(ts[2:],np.diff(ass)/np.diff(ts[1:]), '.',)
                ax.scatter(vs.values[2:],np.diff(ass)/np.diff(ts[1:]),  alpha = .6)
        ax.grid('on')
        #
        #
        return axs
        
    

    @classmethod
    def plot_opensim_lateraldynamics(self, adapted_ts_paths, followerssim_paths_vehs, axs = False, leaderstates = False, figsize = (5,3), alpha = .4, vmax = 200/3.6, quantiles = [.5], loc_idx = 2, v_idx = 3, n_acceleration_plotted = False, N_plotted = 40):
        """
        
        plot the simulation on open road. 
        
        -------------------------------
        @input: followerssim_paths_vehs
        
            followerssim_paths_vehs[path_idx][veh_idx] is a pd. shape is (6, moments_N)
        
        @input: adapted_ts_paths
        
            adapted_ts = adapted_ts_paths[path_idx]
        
            adapted_ts is a list contaiing the lists. 
            
        """
        
        if isinstance(axs, bool):
            fig,axs = plt.subplots(figsize = figsize, ncols = 2, nrows = 2)
            #ax.set_xlabel('Time (sec)');ax.set_ylabel('x (m)'); 
            #ax.grid();
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        #
        #####################
        ax = axs[0,0]
        ax.set_xlabel('Time (sec)');ax.set_ylabel('y (m)'); 
        i=0
        for path_idx in followerssim_paths_vehs.keys():
            i=i+1
            if i>=N_plotted:break
            for veh_idx in followerssim_paths_vehs[path_idx].keys():
                ys = followerssim_paths_vehs[path_idx][veh_idx].iloc[loc_idx, :]
                ax.plot(adapted_ts_paths[path_idx], ys.values, alpha = alpha)
        ax.grid('on')
        
        #####################v
        ax = axs[0, 1]
        ax.set_xlabel('Time (sec)');ax.set_ylabel('vy (m/s)'); 
        i=0
        for path_idx in followerssim_paths_vehs.keys():
            i=i+1
            if i>=N_plotted:break
            for veh_idx in followerssim_paths_vehs[path_idx].keys():
                ys = followerssim_paths_vehs[path_idx][veh_idx].iloc[v_idx, :]
                ax.plot(adapted_ts_paths[path_idx], ys.values, alpha = alpha)
        ax.grid('on')
        #####################acceleration
        ax = axs[1, 0]
        ax.set_xlabel('Time (sec)');ax.set_ylabel('$accelerations (m/s^2)$'); 
        i=0
        for path_idx in followerssim_paths_vehs.keys():
            i=i+1
            if i>=N_plotted:break
            for veh_idx in followerssim_paths_vehs[path_idx].keys():
                vs = followerssim_paths_vehs[path_idx][veh_idx].iloc[v_idx, :]
                #
                ax.plot(adapted_ts_paths[path_idx][1:], np.diff(vs.values)/np.diff(adapted_ts_paths[path_idx]), alpha = alpha)
        ax.grid('on')
        #####################jerk
        ax = axs[1, 1]
        ax.set_xlabel('Speed (m/s)');ax.set_ylabel('$jerk (m/s^{3}}))$'); 
        for path_idx in followerssim_paths_vehs.keys():
            for veh_idx in followerssim_paths_vehs[path_idx].keys():
                xvs = followerssim_paths_vehs[path_idx][veh_idx].iloc[1, :]
                #
                vs = followerssim_paths_vehs[path_idx][veh_idx].iloc[v_idx, :]
                ass = np.diff(vs.values)/np.diff(adapted_ts_paths[path_idx])
                jerks = np.diff(ass)/np.diff(adapted_ts_paths[path_idx][1:])
                #
                ax.plot(xvs.values[2:], jerks, '.',  alpha = .6)
        ax.grid('on')
        #
        #
        return axs
    
    @classmethod
    def plot_opensim_leaderstates_followerpaths(self, leaderstates, followerssim_paths, ax = False, figsize = (5,3), alpha = .4, vmax = 200/3.6, quantiles = [.5], loc_idx = 0, v_idx = 1):
        """
        
        plot the simulation on open road. 
        
        -------------------------------
        
        @input: leaderstates
            
            leaderstates[moment] is a 1d array. [x, vx, y, vy, zlon, zlat]
            
            leaderstates = irre.SingleLaneSim.LeaderVehicleState_constancespeed(ts = ts, v_ms = v/3.6)
        
        
        @input: followersim_paths
            
            followerssim_paths[veh_idx][v][path_idx] = pd, columns are moments, which is the same as input arg ts.
            
            followersim_paths = irre.SingleLaneSim.sim_openroad_pure_HDVs(ts = ts, init_states = init_states, leader_states = leaderstates,  \
                    N_paths = 200, tao_delay = .8, acc_model = 'IDM', diffusion_typee = 'speed_dependent_jerk')
            
        """
        
        if isinstance(ax, bool):
            fig,axs= plt.subplots(figsize = figsize, ncols = 2)
            #ax.set_xlabel('Time (sec)');ax.set_ylabel('x (m)'); 
            #ax.grid();
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        #
        #####################x
        ax = axs[0]
        ax.set_xlabel('Time (sec)');ax.set_ylabel('x (m)'); 
        xs_leader = np.array([leaderstates[t][loc_idx] for t in sorted(leaderstates.keys())])
        ax.plot(sorted(leaderstates.keys()), xs_leader, '.-k', alpha = alpha, )
        #
        #
        for veh_idx in followerssim_paths.keys():
            for path_idx in followerssim_paths[veh_idx].keys():
                #
                xs = followerssim_paths[veh_idx][path_idx].iloc[loc_idx, :]
                #
                ax.plot(xs.index, xs.values, alpha = alpha)
        #####################v
        ax = axs[1]
        ax.set_xlabel('Time (sec)');ax.set_ylabel('v (m/s)'); 
        vs_leader = np.array([leaderstates[t][v_idx] for t in sorted(leaderstates.keys())])
        ax.plot(sorted(leaderstates.keys()), vs_leader, '.-k', alpha = alpha, )
        #
        #
        for veh_idx in followerssim_paths.keys():
            for path_idx in followerssim_paths[veh_idx].keys():
                #
                vs = followerssim_paths[veh_idx][path_idx].iloc[v_idx, :]
                #
                ax.plot(vs.index, vs.values, alpha = alpha)
        
        return ax
        

    
    @classmethod
    def plot_opensim_leaderstates_followerpaths_relative(self, leaderstates, followerssim_paths, ax = False, figsize = (5,3), alpha = .4, vmax = 200/3.6, quantiles = [.5]):
        """
        
        plot the simulation on open road. 
        
        -------------------------------
        
        @input: leaderstates
            
            leaderstates[moment] is a 1d array. [x, vx, y, vy, zlon, zlat]
            
            leaderstates = irre.SingleLaneSim.LeaderVehicleState_constancespeed(ts = ts, v_ms = v/3.6)
        
        
        @input: followersim_paths
            
            followerssim_paths[veh_idx][path_idx] = pd, columns are moments, which is the same as input arg ts.
            
            followersim_paths = irre.SingleLaneSim.sim_openroad_pure_HDVs(ts = ts, init_states = init_states, leader_states = leaderstates,  \
                    N_paths = 200, tao_delay = .8, acc_model = 'IDM', diffusion_typee = 'speed_dependent_jerk')
            
        """
        
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)
            ax.set_xlabel('Time (sec)');ax.set_ylabel('x (m)'); 
            #ax.grid();
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        #
        xs_leader = np.array([leaderstates[t][0] for t in sorted(leaderstates.keys())])
        
        
        #
        for veh_idx in followerssim_paths.keys():
            for path_idx in followerssim_paths[veh_idx].keys():
                #
                xs = followerssim_paths[veh_idx][path_idx].iloc[0, :]
                #
                ax.plot(xs.index, xs_leader-xs.values, alpha = alpha)
        
        return ax
        
        
    
    @classmethod
    def plot_followerpaths(self, followerssim_paths, ax = False, figsize = (5,3), alpha = .4, vmax = 200/3.6):
        """
        
        plot the simulation on open road. 
        
        -------------------------------
        
        @input: leaderstates
            
            leaderstates[moment] is a 1d array. [x, vx, y, vy, zlon, zlat]
            
            leaderstates = irre.SingleLaneSim.LeaderVehicleState_constancespeed(ts = ts, v_ms = v/3.6)
        
        
        @input: followersim_paths
            
            followerssim_paths[veh_idx][path_idx] = pd, columns are moments, which is the same as input arg ts.
            
            followersim_paths = irre.SingleLaneSim.sim_openroad_pure_HDVs(ts = ts, init_states = init_states, leader_states = leaderstates,  \
                    N_paths = 200, tao_delay = .8, acc_model = 'IDM', diffusion_typee = 'speed_dependent_jerk')
            
        """
        
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)
            ax.set_xlabel('Time (sec)');ax.set_ylabel('x (m)'); 
            #ax.grid();
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        #
        xs_leader = np.array([leaderstates[t][0] for t in sorted(leaderstates.keys())])
        for veh_idx in followerssim_paths.keys():
            for path_idx in followerssim_paths[veh_idx].keys():
                #
                xs = followerssim_paths[veh_idx][path_idx].iloc[0, :]
                #
                ax.plot(xs.index, xs_leader-xs.values, alpha = alpha)
        
        return ax
        
    
    @classmethod
    def plot_speed_color(self, STATES, ax = False, figsize = (5,3), alpha = .4, vmax = 200/3.6):
        """
        
        @input: STATES
        
            STATES[moment] = {vid:np.array([x,vx, y, vy, zlon, zlat])}
            
            
        """
        
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)
            ax.set_xlabel('Time (sec)');ax.set_ylabel('x (m)'); 
            #ax.grid();
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        for t in sorted(STATES.keys()):
            #SNAPSHOTS[t].keys are the vids
            vids = set(STATES[t].keys())
            xs = [STATES[t][vid][0] for vid in vids]
            #make color
            vxs = [STATES[t][vid][1] for vid in vids]
            vxs1 = list(np.array(vxs)/vmax)
            vxs2 = [np.array([v, v, 0]) for v in vxs1]
            #print(max(vxs1))
            #
            ts = np.array([t]*len(xs))
            
            ax.scatter(ts, xs, c = np.tan(vxs1), marker = '.')
        
        #
        #ax.set_xlabel('x ( m )');ax.set_ylabel('y ( m )'); ax.grid()
        
        #plt.colorbar()
        plt.tight_layout()
        
        return ax
    
    @classmethod
    def plot_sim(self, STATES, ax = False, figsize = (5,3), alpha = .4,):
        """
        
        @input: STATES
        
            STATES[moment] = {vid:np.array([x,vx, y, vy, zlon, zlat])}
        
        
        """
        
        
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)
            ax.set_xlabel('x');ax.set_ylabel('y'); 
            #ax.grid();
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        
        #vids_startmoments[vid] = t, means that the vid first appear at moment t. 
        vids_startmoments = {}
        vids_existing = set()
        for t in sorted(STATES.keys()):
            #SNAPSHOTS[t].keys are the vids
            vids = set(STATES[t].keys())
            #
            #find the 
            new_vids = vids.difference(vids_existing)
            for vid in new_vids:
                vids_startmoments[vid] = t
                
            vids_existing = vids
        
        #
        for vid in vids_startmoments:
            #
            t = vids_startmoments[vid]
            #find the keys (effective_ts) that vid have data. 
            tmp = np.array(sorted(STATES.keys()))
            effective_ts =tmp[tmp>=t]
            
            #
            xs = [STATES[t][vid][0] for t  in effective_ts]
            ys = [STATES[t][vid][2] for t  in effective_ts]
            ax.plot(xs, ys, '.', alpha = alpha)
            #ax = self.plot_path(path = sim_res, ax = ax, figsize = figsize, alpha = alpha,)
        
        #
        ax.set_xlabel('x ( m )');ax.set_ylabel('y ( m )'); ax.grid()
        
            
        plt.tight_layout()
        
        return ax
    
    
    def plot_lanemarks_boundary(self, ax = False, figsize = (5,3), alpha = .4, markcolor = 'y', boundarycolor = 'k'):
        """
        
        """
        #==========================================
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)
            ax.set_xlabel('x');ax.set_ylabel('y'); 
            #ax.grid();
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        
        #
        ax.plot([0, self.length], (self.road_bounds[0],  self.road_bounds[0]), 'k.-', linewidth = 4 )
        ax.plot([0, self.length], (self.road_bounds[1],  self.road_bounds[1]), 'k.-', linewidth = 4 )
        
        return ax

    @classmethod
    def plot_sim_from_snapshots(self, snapshots, ax = False, figsize = (5,3), alpha = .4,):
        """
        
        state variable is self.vehs_dict[vehicle_id] = [x, vx, y, vy, zlon, zlat]
        
        ------------------------------------------------
        @input: STATES
        
            STATES[moment] = {vid:np.array([x,vx, y, vy, zlon, zlat])}

        @input: snapshots
                
                #
                self.snapshots[T]['vehs_at_lanes'] = copy.deepcopy(self.vehs_at_lanes)
                self.snapshots[T][vid] = {'S':copy.deepcopy(self.vehs_dict[vid]), \
                    'feasibleincrement':copy.deepcopy(feasible_increment), \
                    'L':copy.deepcopy(L_dicts[vid]), \
                    'F':copy.deepcopy(F_dicts[vid]), \
                    'potentials':copy.deepcopy(potentials_dict[vid]),}
        
        """
        
        
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)
            ax.set_xlabel('x');ax.set_ylabel('y'); 
            #ax.grid();
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        
        #vids_startmoments[vid] = t, means that the vid first appear at moment t. 
        t0 = max(snapshots.keys())
        vids = snapshots[t0]['vehs_at_lanes']
        """
        vids_startmoments = {}
        vids_existing = set()
        for t in sorted(snapshots.keys()):
            #SNAPSHOTS[t].keys are the vids
            vids = snapshots[t]['vehs_at_lanes']
            #
            #find the 
            new_vids = vids.difference(vids_existing)
            for vid in new_vids:
                vids_startmoments[vid] = t
                
            vids_existing = vids
        
        """
        
        #
        for vid in vids:
            #
            #t = vids_startmoments[vid]
            #find the keys (effective_ts) that vid have data. 
            ts = np.array(sorted(snapshots.keys()))
            effective_ts =  [t for t in ts if vid in snapshots[t]['vehs_at_lanes']]#tmp[tmp>=t]
            
            #
            #print(snapshots[0][vid].keys())
            xs = [snapshots[t][vid]['S'][0] for t in effective_ts]
            ys = [snapshots[t][vid]['S'][2] for t in effective_ts]
            ax.plot(xs, ys, '.', alpha = alpha)
            #ax = self.plot_path(path = sim_res, ax = ax, figsize = figsize, alpha = alpha,)
        
        #
        ax.set_xlabel('x ( m )');ax.set_ylabel('y ( m )'); ax.grid()
        
            
        plt.tight_layout()
        
        return ax

    @classmethod
    def plot_sim_marginal_distribution(self, STATES, roadlength, ax = False, figsize = (5,3), alpha = .4, n_xs_marginal = 10, tolerance_x_find_y = 1.0, bins = 20):
        """
        plot the xy and the p(y | x), i.e. the marginal distribution. 
        
        
        @input: STATES
        
            STATES[moment] = {vid:np.array([x,vx, y, vy, zlon, zlat])}
        
        @input: tolerance_x_find_y
        
            given x0, when finding y, the method is that:
            
                - find the idx that with x sasitf abs(x-x0)<tolerance_x_find_y
                - get the y. 
        
        @input: n_xs_marginal
        
            the number of the marginals to be calculated and plotted. 
        
        """
        #
        xs_marginals  = np.linspace(1, roadlength-1, n_xs_marginal)
        
        if isinstance(ax, bool):
            fig,axs = plt.subplots(figsize = figsize, ncols = 1, nrows = 2)
            #ax.set_xlabel('x');ax.set_ylabel('y'); 
            #ax.grid();
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        ##############################First plot
        ax = axs[0]
        #vids_startmoments[vid] = t, means that the vid first appear at moment t. 
        vids_startmoments = {}
        vids_existing = set()
        for t in sorted(STATES.keys()):
            #SNAPSHOTS[t].keys are the vids
            vids = set(STATES[t].keys())
            #
            #find the 
            new_vids = vids.difference(vids_existing)
            for vid in new_vids:
                vids_startmoments[vid] = t
            vids_existing = vids
        #
        for vid in vids_startmoments:
            #
            t = vids_startmoments[vid]
            #find the keys (effective_ts) that vid have data. 
            tmp = np.array(sorted(STATES.keys()))
            effective_ts =tmp[tmp>=t]
            
            #
            xs = [STATES[t][vid][0] for t  in effective_ts]
            ys = [STATES[t][vid][2] for t  in effective_ts]
            ax.plot(xs, ys, '.', alpha = alpha)
            #ax = self.plot_path(path = sim_res, ax = ax, figsize = figsize, alpha = alpha,)
        
        ax.set_ylim([0, 3.5])
        ax.set_xlabel('x ( m )');ax.set_ylabel('y ( m )'); ax.grid()
        
        #############################second plot
        ax = axs[1]
        for x in xs_marginals:
            ys = []
            for t in sorted(STATES.keys()):
                for vid in STATES[t].keys():
                    if abs(STATES[t][vid][0] - x)<=tolerance_x_find_y:
                        ys.append(STATES[t][vid][2])
            #
            hist, edges = np.histogram(ys, bins = bins)
            ax.plot(edges[1:], hist/sum(hist)/(edges[-1] - edges[-2]), label = 'loc = ' + str(int(x*100)/100.0) + ' m')
        ax.set_xlim([0, 3.5])
        ax.legend()
        ax.set_xlabel('lateral location y ( m )')
        ax.set_ylabel('probability')
        ax.grid()
        
        plt.tight_layout()
        
        return axs
    
    
    @classmethod
    def plot_speed_profile_3D_singlelane(self, snapshots, ax = False, figsize = (5,3), alpha = .4,):
        """
        plot the speed profile of 3D, i.e. (t, x, v)
        
        
        @input: 
        
                #SNAPSHOT
                self.snapshots[T]['vehs_at_lanes'] = copy.deepcopy(self.vehs_at_lanes)
                self.snapshots[T][vid] = {'S':copy.deepcopy(self.vehs_dict[vid]), \
                    #'feasibleincrement':copy.deepcopy(feasible_increment), \
                    'L':copy.deepcopy(L_dicts[vid]), \
                    'F':copy.deepcopy(F_dicts[vid]), \
                    'vehs_info_within_area_left':copy.deepcopy(vehs_info_within_area_left_dict[vid]), \
                    'vehs_info_within_area_right':copy.deepcopy(vehs_info_within_area_right_dict[vid]), \
                    'potentials':copy.deepcopy(potentials_dict[vid]), }
                #
        """
        
        #from mpl_toolkits.mplot3d.axes3d import get_test_data
        # This import registers the 3D projection, but is otherwise unused.
        #from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
        
        t = sorted(snapshots.keys())[0]
        #lanes_ids = snapshots[t]['vehs_at_lanes'].keys()
        #lanes_number = len(snapshots[t]['vehs_at_lanes'])
        #==========================================
        if isinstance(ax, bool):
            #fig, ax = plt.subplots(figsize = figsize, ncols = 1, nrows = 1, projection='3d')
            
            ax = plt.figure().add_subplot(projection='3d')
            #ax.set_xlabel('t');ax.set_ylabel('x');ax.set_zlabel('speed'); 
            #ax.grid();
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        #
        #--------------------------------------
        #fig = plt.figure(figsize=plt.figaspect(0.5))
        #fig = plt.figure(figsize=figsize)
        Ts = np.array(sorted(snapshots.keys()))
        #for laneidx,laneid in enumerate(lanes_ids):
        #for laneidx in range(lanes_number):
        #ax = axs[laneidx]
        #ax = fig.add_subplot(lanes_number, 1, laneidx, projection='3d')
        #
        for t in Ts:
            #SNAPSHOTS[t].keys are the vids
            vids = snapshots[t].keys()#set(snapshots[t].keys())
            #print(laneidx, vids)
            xs = [snapshots[t][vid]['S'][0] for vid in vids]
            #make color
            vxs = [snapshots[t][vid]['S'][1] for vid in vids]
            #vxs1 = list(np.array(vxs)/vmax)
            #vxs2 = [np.array([v, v, 0]) for v in vxs1]
            #print(max(vxs1))
            #
            ts = np.array([t]*len(xs))
            
            
            ax.plot(ts, xs, vxs,  marker = '.')
        ax.set_xlabel('Time');ax.set_ylabel('x (m)');ax.set_zlabel('speed');



    @classmethod
    def plot_twodim_trajectories_givensnapshots_singlelane(self, snapshots, length = 500, ax = False, figsize = (5,3), alpha = .4, vmax = 200/3.6, timeintervalploted = (100, 200), cmap = 'Blues', N_plotted = 20):
        """
        plot the speed profile of 3D, i.e. (t, x, v)
        
        {vid:np.array([x,vx, y, vy, zlon, zlat])}
        
        
        @input: cmap
        
        ValueError: 'OrangeBlue' is not a valid value for name; supported values are 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r'
        
        @input: snapshots
        
                #
                self.snapshots[T]['vehs_at_lanes'] = copy.deepcopy(self.vehs_at_lanes)
                self.snapshots[T][vid] = {'S':copy.deepcopy(self.vehs_dict[vid]), \
                    'feasibleincrement':copy.deepcopy(feasible_increment), \
                    'L':copy.deepcopy(L_dicts[vid]), \
                    'F':copy.deepcopy(F_dicts[vid]), \
                    'potentials':copy.deepcopy(potentials_dict[vid]),}
                
        """
        from matplotlib.collections import LineCollection
        #from mpl_toolkits.mplot3d.axes3d import get_test_data
        # This import registers the 3D projection, but is otherwise unused.
        #from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
        
        t0 = sorted(snapshots.keys())[0]
        #lanes_ids = snapshots[t]['vehs_at_lanes'].keys()
        #lanes_number = len(snapshots[t]['vehs_at_lanes'])
        #==========================================
        if isinstance(ax, bool):
            #fig, ax = plt.subplots(figsize = figsize, ncols = 1, nrows = 1, projection='3d')
            fig, ax = plt.subplots(figsize = figsize)
            ax.set_xlabel('x');ax.set_ylabel('y'); 
            #ax = plt.figure(figsize = figsize).add_subplot(projection='3d')
            #ax.set_xlabel('t');ax.set_ylabel('x');ax.set_zlabel('speed'); 
            #ax.grid();
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        #
        #--------------------------------------
        #fig = plt.figure(figsize=plt.figaspect(0.5))
        #fig = plt.figure(figsize=figsize)
        TS = np.array(sorted(snapshots.keys()))
        #for laneidx,laneid in enumerate(lanes_ids):
        #for laneidx in range(lanes_number):
        #ax = axs[laneidx]
        #ax = fig.add_subplot(lanes_number, 1, laneidx, projection='3d')
        #
        #vids = snapshots[t0]['vehs_at_lanes'][laneid]#set(snapshots[t].keys())
        vids0 = []
        for t in TS:
            vids0.extend(snapshots[t]['vehs_at_lanes'])
        vids = set(vids0)
        #
        #
        TS_es = []
        XS_es = []
        YS_es = []
        VXs1  = []
        for vid in vids:
            color = np.random.uniform(size = (3,))
            #Note that the state is [x, vx, y, vy, zlon, zlat]
            ts0 = [t for t in TS if vid in snapshots[t]]
            xs0 = [snapshots[t][vid]['S'][0] for t in TS if vid in snapshots[t]]
            ys0 = [snapshots[t][vid]['S'][2] for t in TS if vid in snapshots[t]]
            #
            vxs = [snapshots[t][vid]['S'][1] for t in TS if vid in snapshots[t]]
            vxs1 = list(np.array(vxs)/vmax)

            #SEE https://stackoverflow.com/questions/64267329/how-to-plot-a-gradient-color-line
            #cut the trajectories into multi segments, becaus ethe period condition. 
            idxs_segments = TwoDimMicroModel.TransformPeriodic_ys_2MultiSegment_return_idxs_segments(ys = xs0, length = length)
            #ts_es,xs_es = TwoDimMicroModel.TransformPeriodic_ys_2MultiSegment(ys = xs0, length = length, ts = TS)
            #
            for segment in idxs_segments:
                TS_es.append(np.array(ts0)[segment])
                #
                XS_es.append(np.array(xs0)[segment])
                YS_es.append(np.array(ys0)[segment])
                #
                VXs1.append(np.array(vxs1)[segment])
        ##################
        N_plotted = min(N_plotted, len(TS_es))
        selected_idxes = np.random.choice(list(range(len(TS_es))), N_plotted)
        #for ts,xs,ys,vs in zip(TS_es,XS_es, YS_es, VXs1):
        for idx in selected_idxes:
            ts = TS_es[idx]
            xs = XS_es[idx]
            ys = YS_es[idx]
            vs = VXs1[idx]
            #
            points = np.array([xs,ys]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            #lc = LineCollection(segments, cmap='viridis', alpha = alpha)
            lc = LineCollection(segments, cmap = cmap, alpha = alpha)
            lc.set_array(vs)
            ax.add_collection(lc)
            
            """
            #cut the trajectories into multi segments, becaus ethe period condition. 
            ts_es,xs_es = TwoDimMicroModel.TransformPeriodic_ys_2MultiSegment(ys = xs0, length = length, ts = TS)
            for ts,xs in zip(ts_es,xs_es):
                #ax.plot(ts,xs, color = color)
                ax.plot(ts,xs, cmap=plt.get_cmap('jet'))
            """
        
        return ax
    


    @classmethod
    def plot_twodim_trajectories_givenSTATES_singlelane(self, STATES, length = 500, ax = False, figsize = (5,3), alpha = .4, vmax = 200/3.6, timeintervalploted = (100, 200), cmap = 'Blues', N_plotted = 20):
        """
        plot the speed profile of 3D, i.e. (t, x, v)
        
        {vid:np.array([x,vx, y, vy, zlon, zlat])}
        
        --------------------------------------------------------------------
        @input: STATES,irregular_vehs_ids
            
            STATES is a dict. 
                
                STATES[moment][vid] is a 1d array, shape is (6,), they are:
                    
                    self.vehs_dict[vehicle_id] = [x, vx, y, vy, zlon, zlat]
            
        
        @input: cmap
            
            ValueError: 'OrangeBlue' is not a valid value for name; supported values are 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r'
        
        @input: snapshots
        
                #
                self.snapshots[T]['vehs_at_lanes'] = copy.deepcopy(self.vehs_at_lanes)
                self.snapshots[T][vid] = {'S':copy.deepcopy(self.vehs_dict[vid]), \
                    'feasibleincrement':copy.deepcopy(feasible_increment), \
                    'L':copy.deepcopy(L_dicts[vid]), \
                    'F':copy.deepcopy(F_dicts[vid]), \
                    'potentials':copy.deepcopy(potentials_dict[vid]),}
                
        """
        from matplotlib.collections import LineCollection
        #from mpl_toolkits.mplot3d.axes3d import get_test_data
        # This import registers the 3D projection, but is otherwise unused.
        #from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
        
        t0 = sorted(STATES.keys())[0]
        #lanes_ids = snapshots[t]['vehs_at_lanes'].keys()
        #lanes_number = len(snapshots[t]['vehs_at_lanes'])
        #==========================================
        if isinstance(ax, bool):
            #fig, ax = plt.subplots(figsize = figsize, ncols = 1, nrows = 1, projection='3d')
            fig, ax = plt.subplots(figsize = figsize)
            ax.set_xlabel('x');ax.set_ylabel('y'); 
            #ax = plt.figure(figsize = figsize).add_subplot(projection='3d')
            #ax.set_xlabel('t');ax.set_ylabel('x');ax.set_zlabel('speed'); 
            #ax.grid();
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        #
        #--------------------------------------
        #fig = plt.figure(figsize=plt.figaspect(0.5))
        #fig = plt.figure(figsize=figsize)
        TS = np.array(sorted(STATES.keys()))
        #for laneidx,laneid in enumerate(lanes_ids):
        #for laneidx in range(lanes_number):
        #ax = axs[laneidx]
        #ax = fig.add_subplot(lanes_number, 1, laneidx, projection='3d')
        #
        #vids = snapshots[t0]['vehs_at_lanes'][laneid]#set(snapshots[t].keys())
        vids = set(STATES[t0].keys())
        #
        #
        TS_es = []
        XS_es = []
        YS_es = []
        VXs1  = []
        for vid in vids:
            color = np.random.uniform(size = (3,))
            #Note that the state is [x, vx, y, vy, zlon, zlat]
            ts0 = [t for t in TS if vid in STATES[t]]
            xs0 = [STATES[t0][vid][0] for t in TS if vid in STATES[t]]
            ys0 = [STATES[t0][vid][2] for t in TS if vid in STATES[t]]
            #
            vxs = [STATES[t0][vid][1] for t in TS if vid in STATES[t]]
            vxs1 = list(np.array(vxs)/vmax)

            #SEE https://stackoverflow.com/questions/64267329/how-to-plot-a-gradient-color-line
            #cut the trajectories into multi segments, becaus ethe period condition. 
            idxs_segments = TwoDimMicroModel.TransformPeriodic_ys_2MultiSegment_return_idxs_segments(ys = xs0, length = length)
            #ts_es,xs_es = TwoDimMicroModel.TransformPeriodic_ys_2MultiSegment(ys = xs0, length = length, ts = TS)
            #
            for segment in idxs_segments:
                TS_es.append(np.array(ts0)[segment])
                #
                XS_es.append(np.array(xs0)[segment])
                YS_es.append(np.array(ys0)[segment])
                #
                VXs1.append(np.array(vxs1)[segment])
        ##################
        N_plotted = len(TS_es)
        selected_idxes = np.random.choice(list(range(len(TS_es))), N_plotted)
        #for ts,xs,ys,vs in zip(TS_es,XS_es, YS_es, VXs1):
        #print(len(selected_idxes))
        for idx in selected_idxes:
            ts = TS_es[idx]
            xs = XS_es[idx]
            ys = YS_es[idx]
            vs = VXs1[idx]
            #
            points = np.array([xs,ys]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            #lc = LineCollection(segments, cmap='viridis', alpha = alpha)
            lc = LineCollection(segments, cmap = cmap, alpha = alpha)
            lc.set_array(vs)
            ax.add_collection(lc)
            
            """
            #cut the trajectories into multi segments, becaus ethe period condition. 
            ts_es,xs_es = TwoDimMicroModel.TransformPeriodic_ys_2MultiSegment(ys = xs0, length = length, ts = TS)
            for ts,xs in zip(ts_es,xs_es):
                #ax.plot(ts,xs, color = color)
                ax.plot(ts,xs, cmap=plt.get_cmap('jet'))
            """
        
        return ax
    

    @classmethod
    def plot_speed_profile_singlelane_trajectories(self, snapshots, length = 500, laneid = 0, ax = False, figsize = (5,3), alpha = .4, vmax = 200/3.6, timeintervalploted = (100, 200), cmap = 'Blues', blacked_trajectory_vids = []):
        """
        plot the speed profile of 3D, i.e. (t, x, v)
        
        @input: blacked_trajectory_vids
            
            the trajectories of these vids will be totally black. 
        
        @input: snapshots
        
                self.snapshots[T][vid] = {'S':copy.deepcopy(self.vehs_dict[vid]), \
                    'feasibleincrement':copy.deepcopy(feasible_increment), \
                    'L':copy.deepcopy(L_dicts[vid]), \
                    'F':copy.deepcopy(F_dicts[vid]), \
                    'potentials':copy.deepcopy(potentials_dict[vid]), }
        
        
        
        @input: cmap
        
        ValueError: 'OrangeBlue' is not a valid value for name; supported values are 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r'
        
        @input: 
        
                #SNAPSHOT
                self.snapshots[T]['vehs_at_lanes'] = copy.deepcopy(self.vehs_at_lanes)
                self.snapshots[T][vid] = {'S':copy.deepcopy(self.vehs_dict[vid]), \
                    #'feasibleincrement':copy.deepcopy(feasible_increment), \
                    'L':copy.deepcopy(L_dicts[vid]), \
                    'F':copy.deepcopy(F_dicts[vid]), \
                    'vehs_info_within_area_left':copy.deepcopy(vehs_info_within_area_left_dict[vid]), \
                    'vehs_info_within_area_right':copy.deepcopy(vehs_info_within_area_right_dict[vid]), \
                    'potentials':copy.deepcopy(potentials_dict[vid]), }
                #
        """
        from matplotlib.collections import LineCollection
        #from mpl_toolkits.mplot3d.axes3d import get_test_data
        # This import registers the 3D projection, but is otherwise unused.
        #from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
        
        t0 = sorted(snapshots.keys())[0]
        #lanes_ids = snapshots[t]['vehs_at_lanes'].keys()
        #lanes_number = len(snapshots[t]['vehs_at_lanes'])
        #==========================================
        if isinstance(ax, bool):
            #fig, ax = plt.subplots(figsize = figsize, ncols = 1, nrows = 1, projection='3d')
            fig, ax = plt.subplots(figsize = figsize)
            #ax = plt.figure(figsize = figsize).add_subplot(projection='3d')
            #ax.set_xlabel('t');ax.set_ylabel('x');ax.set_zlabel('speed'); 
            #ax.grid();
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        #
        #--------------------------------------
        #fig = plt.figure(figsize=plt.figaspect(0.5))
        #fig = plt.figure(figsize=figsize)
        TS = np.array(sorted(snapshots.keys()))
        #for laneidx,laneid in enumerate(lanes_ids):
        #for laneidx in range(lanes_number):
        #ax = axs[laneidx]
        #ax = fig.add_subplot(lanes_number, 1, laneidx, projection='3d')
        #
        #vids = snapshots[t0]['vehs_at_lanes'][laneid]#set(snapshots[t].keys())
        vids0 = []
        for t in TS:vids0.extend(snapshots[t].keys())
        vids = set(vids0)
        #
        ####################################################REGULAR VEHICLES
        TS_es = []
        XS_es = []
        VXs1  = []
        for vid in vids:
            color = np.random.uniform(size = (3,))
            #
            #print(snapshots[t][vid])
            xs0 = [snapshots[t][vid]['S'][0] for t in TS if vid in snapshots[t]['vehs_at_lanes']]
            #
            vxs = [snapshots[t][vid]['S'][1] for t in TS if vid in snapshots[t]['vehs_at_lanes']]
            vxs1 = list(np.array(vxs)/vmax)

            #SEE https://stackoverflow.com/questions/64267329/how-to-plot-a-gradient-color-line
            #cut the trajectories into multi segments, becaus ethe period condition. 
            idxs_segments = TwoDimMicroModel.TransformPeriodic_ys_2MultiSegment_return_idxs_segments(ys = xs0, length = length)
            #ts_es,xs_es = TwoDimMicroModel.TransformPeriodic_ys_2MultiSegment(ys = xs0, length = length, ts = TS)
            #
            for segment in idxs_segments:
                TS_es.append(np.array(TS)[segment])
                XS_es.append(np.array(xs0)[segment])
                VXs1.append(np.array(vxs1)[segment])
        ##################
        for ts,xs,vs in zip(TS_es,XS_es, VXs1):
            points = np.array([ts, xs]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            #lc = LineCollection(segments, cmap='viridis', alpha = alpha)
            lc = LineCollection(segments, cmap = cmap, alpha = alpha)
            lc.set_array(vs)
            ax.add_collection(lc)
            
            """
            #cut the trajectories into multi segments, becaus ethe period condition. 
            ts_es,xs_es = TwoDimMicroModel.TransformPeriodic_ys_2MultiSegment(ys = xs0, length = length, ts = TS)
            for ts,xs in zip(ts_es,xs_es):
                #ax.plot(ts,xs, color = color)
                ax.plot(ts,xs, cmap=plt.get_cmap('jet'))
            """
            #ax.plot(TS, xs0)
        
        ####################################################BLACKED VEHICLES
        TS_es = []
        XS_es = []
        VXs1  = []
        for vid in blacked_trajectory_vids:
            #
            #print(snapshots[t][vid])
            xs0 = [snapshots[t][vid]['S'][0] for t in TS if vid in snapshots[t]['vehs_at_lanes']]
            #
            vxs = [snapshots[t][vid]['S'][1] for t in TS if vid in snapshots[t]['vehs_at_lanes']]
            vxs1 = list(np.array(vxs)/vmax)

            #SEE https://stackoverflow.com/questions/64267329/how-to-plot-a-gradient-color-line
            #cut the trajectories into multi segments, becaus ethe period condition. 
            idxs_segments = TwoDimMicroModel.TransformPeriodic_ys_2MultiSegment_return_idxs_segments(ys = xs0, length = length)
            #ts_es,xs_es = TwoDimMicroModel.TransformPeriodic_ys_2MultiSegment(ys = xs0, length = length, ts = TS)
            #
            for segment in idxs_segments:
                TS_es.append(np.array(TS)[segment])
                XS_es.append(np.array(xs0)[segment])
                VXs1.append(np.array(vxs1)[segment])
                #ax.plot(np.array(TS)[segment], np.array(xs0)[segment], 'k')
        ##################
        for ts,xs,vs in zip(TS_es,XS_es, VXs1):
            ax.plot(ts, xs, 'k')
        
        #fig.colorbar(ax=ax)
        ax.set_xlabel('Time');ax.set_ylabel('x (m)');#ax.set_zlabel('speed');
        return ax

    @classmethod
    def equilibeium_v_IDM(self, density_veh_per_km = 100, idm_paras = idm_paras):
        """
        given the density, find the equilibrium speed,. 
        The model is IDM. 
        
        idm_paras = {'idm_vf':120.0/3.6, 'idm_T':1.5, 'idm_delta':4.0, 'idm_s0':2.0, 'idm_a':1.0, 'idm_b':3.21, 'a_MAX':3.5, 'veh_len':5}
        -------------------------------------------------
        @OUTPUT: v
            
            unit is km/h. 
        
        """
        
        pass
    
    
    @classmethod
    def equilibeium_k_IDM(self, v_m_per_sec, l = 5):
        """
        given the speed, find the equilibrium density. 
        
        idm_paras = {'idm_vf':120.0/3.6, 'idm_T':1.5, 'idm_delta':4.0, 'idm_s0':2.0, 'idm_a':1.0, 'idm_b':3.21, 'a_MAX':3.5, 'veh_len':5}
        
        
        
        """
        
        tmp0 = idm_paras['idm_s0']  + v_m_per_sec*idm_paras['idm_T']
        #
        tmp1 = 1-((v_m_per_sec)/(idm_paras['idm_vf'] ))**idm_paras['idm_delta']
        #
        return np.sqrt(tmp1)/tmp0
        #
        
        return 1.0/tmp0/np.power(tmp1, -0.5)
        
        
        
        pass
    
    
    @classmethod
    def LeaderVehicleState_constancespeed(self, v_ms = 10, init_state = np.array([.0, 10, .0, .0, .00001, .00001]), idm_paras = idm_paras,  ts = np.linspace(0, 1000, 2000)):
        """
        Generate the state of the leader vehicle which is assumed as constant speed. 
        
        -----------------------------------------------------------------------
        @input: init_state
        
            the state of the vehicle, init_state = [x, vx, y, vy, zlon, zlat]
            
            units are both m, m/s. 
        
        @input: diffusion_term and jump_term
        
            both are bools. They describe the existence of the 
            
        @OUTPUT: leaderstates
        
            a dict. 
            
            state = leaderstates[idx]
            
            state is a 1d array. [x, vx, y, vy, zlon, zlat]
            
        """
        leaderstates = {}
        leaderstates[min(ts)] = init_state
        #
        for previous_t,t,deltat in zip(ts[:-1], ts[1:], np.diff(ts)):
            #
            previous_state  = leaderstates[previous_t]
            #
            x = v_ms*deltat + previous_state[0]
            #
            new_state = copy.deepcopy(init_state)
            new_state[0] = x
            
            leaderstates[t] = copy.deepcopy(new_state)
        
        return leaderstates
    
    
    @classmethod
    def jump_term_L(self, state, typee = 'constant', idm_paras = idm_paras, OV_parameters = OV_paras):
        """
        state = [x, vx, y , vy, zlon, zlat]
        The jump term. 
        
        'constant','parabolic','OV_Ngoduy','speed_dependent_jerk'
        
        """
        x,vx,y,vy,zlon,zlat = state
        #
        if isinstance(typee, bool):
            return .0
        
        if typee=='constant':
            #
            return 1.0
        elif typee=='parabolic':
            #
            return 1.0*vx*(idm_paras['idm_vf']-vx)/(idm_paras['idm_vf']**2)
            
        elif typee=='OV_Ngoduy':
            
            return OV_paras['sigma0_Ngoduy']*np.sqrt(max(0, vx))
            
        elif typee=='speed_dependent_jerk':
            
            return 1.0*TwoDimMicroModel.SpeedDependentJumpAdjust(max(0, vx))
    
    
    @classmethod
    def diffusion_L(self, state, typee = 'constant', idm_paras = idm_paras, OV_parameters = OV_paras):
        """
        state = [x, vx, y , vy, zlon, zlat]
        
        
        """
        x,vx,y,vy,zlon,zlat = state
        if isinstance(typee, bool):
            return .0
        #
        if typee=='constant':
            #
            return .1
        elif typee=='parabolic':
            #
            return .5*vx*(idm_paras['idm_vf']-vx)/(idm_paras['idm_vf']**2)
            
        elif typee=='OV_Ngoduy':
            
            return OV_paras['sigma0_Ngoduy']*np.sqrt(max(0, vx))
            
        elif typee=='speed_dependent_jerk':
            
            return .1*TwoDimMicroModel.SpeedDependentJumpAdjust(max(0, vx))

        elif typee=='speed_dependent_jerk_Bouadi':
            #Stochastic factors and string stability of traffic flow: Analytical investigation and numerical study based on car-following models
            #dv = f_IDM() * dt + sigma*np.sqrt(V) dW

            return np.sqrt(vx)

    @classmethod
    def sim_openroad_pure_HDVs_withjumps(self, adapted_ts, jumpsizes_adapted_vehs_paths, \
            init_states, leader_states, \
            N_paths = 100, tao_delay = .8, \
            idm_paras = idm_paras, \
            diffusion_typee = 'constant', \
            acc_model = 'IDM', \
            jump_term_type = 'constant', \
            OV_parameters = OV_paras, \
            jerk_constraint = False, two_dim_paras = two_dim_paras):
        """
        Simulation of the open road (non periodic).
        
            sim_paths = sim_openroad_pure_HDVs(ts, init_states, leader_states,  N_paths = 100, tao_delay = .8, diffusion_L = diffusion_L)
        
        --------------------------------------------------------------------
        @input: jump_term_type
        
            may be False, 'constant','parabolic','OV_Ngoduy','speed_dependent_jerk'
            
        @input: adapted_ts, jumpsizes_adapted_vehs_paths
        
            adapted_ts is a list contaiing the lists. 
            
            jumpsizes_adapted = jumpsizes_adapted_vehs_paths[veh_idx][path_idx],is either 0 or a jumptize. 
            
            Generation methods:
            
            ###############Get adapted_ts, jump_or_nots_lists,jumpsizes_adapted_lists###############################################
            reload(irre)
            T_horizon_sec = 1000
            #
            #Number of vehicles that is irregular. 
            N_irregulars = 5
            #
            #distributions of the jump sizes. 
            args_jumpsize_diss = [{'mu':.0, 'sigma': 1.0} for i in range(N_irregulars)]
            jumpdensities =  [2 for i in range(N_irregulars)]
            #
            #
            jumptimes_es,jumpsizes_es = irre.JumpDiffusionDrivingBehavior.JumpMomentsWithJumpSizes_multiplevehicle(T_horizon_sec= T_horizon_sec, args_jumpsize_diss = args_jumpsize_diss, jumpdensities = jumpdensities)
            #
            #determine the ts (a grid) and the adapted_ts
            ts = np.linspace(0, T_horizon_sec, int(T_horizon_sec*2))
            adapted_ts,jump_or_nots_lists,jumpsizes_adapted_lists = irre.JumpDiffusionDrivingBehavior.AdaptEuler(ts = ts, \
                                            jumptimes_lists = jumptimes_es, jumpsizes_lists = jumpsizes_es, )
            
            ##############################################################
        @input: jerk_constraint
        
            a bool. Used to indicate that whether the max and min jerk constraint is applied. 
            
                influence the function, extra input args:
                
                    self.TrimUpdate(jerk_constraint = False, old_old_state = False, old_delta_sec = False)
                
            
                minn,maxx = TwoDimMicroModel.Speed_upperlower_given_jerk_upperlower(v0, v1, deltat0, delta1, )
        
        @input: acc_model
        
            the acceleration model used. 
            
            It can be: 'IDM', 'OV_Ngoduy', 'OV_peng'
            
            OV_peng: 
            
        @input: diffusion_typee = 'constant'
        
            used in self.diffusion_L
            
            can be 'constant', 'parabolic', 'lognormal', 'speed_dependent_jerk', 'speed_dependent_jerk_Bouadi'
            
        @input: init_states
        
            a list of states containing the intiail state of the vehicles, thery are the 1st folloer, second follower...
            
            state = init_states[veh_idx], state = [x, vx, y, vy, zlon, zlat]
        
        @input: diffusion_L
        
            a function that represent the diffusion term in the model
        
        @input: leader_states
        
            an dict of 1darray.
             
            leader_states[time] = veh_state, each state is [x, vx, y, vy, zlon, zlat]
            
            It can be obtained via:
                
                #-------------
                leaderstates = irre.SingleLaneSim.LeaderVehicleState_constancespeed(ts = adapted_ts, v_ms = 40)
        
        @input: vehs_types
        
            a list of the vehicles types. 
        
        @OUTPUT: sim_paths
            
            sim_paths[veh_idx][path_idx] = pd, columns are moments, which is the same as input arg ts.
            
            veh_paths = sim_path[idx_veh] is a dict. 
            veh_path = veh_paths[path_idx]
            
                is a pd. Columns are moments, which is consistemt with input ts. 
        
        """
        #==========Check the moment is correct. 
        if len(adapted_ts)!=len(leader_states):
            raise ValueError('sdsdfsdfsadf')
        #========Check N_paths
        someveh_idx = np.random.choice(list(jumpsizes_adapted_vehs_paths.keys()))
        if len(jumpsizes_adapted_vehs_paths[someveh_idx])!=N_paths:
            raise ValueError('sdsdfsdfsadf')
        
        #
        #sim_paths[veh_idx][path_idx][moment] = state, state = [x, vx,y, vy, zlon, zlat]
        sim_paths = {veh_idx:{pathidx:{min(adapted_ts):init_states[veh_idx]} for pathidx in range(N_paths)} for veh_idx in range(len(init_states))}
        #
        for path_idx in range(N_paths):
            #for time_idx,(t0,t1) in enumerate(zip(adapted_ts[:-1], adapted_ts[1:])):
            tmp = [min(adapted_ts)-np.diff(adapted_ts)[0]] + list(adapted_ts[:-2])
            for time_idx,(t0,t1,t2) in enumerate(zip(tmp, adapted_ts[:-1], adapted_ts[1:])):
                #
                delta_0 = t1 - t0
                deltat = t2 - t1
                #
                for veh_idx in range(len(init_states)):
                    #=========FIND ego state and the leader state at certain moment earlier. 
                    #print(time_idx, t1)
                    ego_state_previous_moment = sim_paths[veh_idx][path_idx][t1]
                    #
                    x,vx,y,vy,zlon,zlat = ego_state_previous_moment
                    #------------find its leader
                    t_leader = t1 - tao_delay
                    #
                    if t_leader<=min(adapted_ts):
                        if veh_idx==0:
                            leader_state = leader_states[min(adapted_ts)]
                        else:
                            leader_state = sim_paths[veh_idx - 1][path_idx][min(adapted_ts)]
                    else:
                        tmp = np.abs(t_leader - np.array(adapted_ts))
                        t_key = np.where(tmp==min(tmp))[0][0]
                        if veh_idx==0:
                            leader_state = leader_states[min(adapted_ts)]
                        else:
                            leader_state = sim_paths[veh_idx - 1][path_idx][min(adapted_ts)]
                    #=========Calculate the deltax
                    deltax = leader_state[0] - ego_state_previous_moment[0]
                    #
                    #========Aceleration,
                    if acc_model=='IDM':
                        acceleration  = TwoDimMicroModel.Potential_IDM(v_self = ego_state_previous_moment[1]*3.6, v_leader = leader_state[1]*3.6, deltax =deltax, idm_paras = idm_paras)
                    elif acc_model=='OV_Ngoduy':
                        acceleration  = TwoDimMicroModel.Potential_OV_Ngoduy(v_self = ego_state_previous_moment[1]*3.6, v_leader = leader_state[1]*3.6, deltax =deltax, OV_paras = OV_paras)
                    elif acc_model=='OV_peng':
                        acceleration  = TwoDimMicroModel.Potential_OV_Peng(v_self = ego_state_previous_moment[1]*3.6, v_leader = leader_state[1]*3.6, deltax =deltax, OV_paras = OV_paras)
                    #
                    F = np.array([vx, acceleration, .0, .0, .0, .0])
                    #
                    #=========Diffusion
                    #brownian = np.random.normal(loc = np.zeros((2,)), scale = deltat_sec)
                    brownian = np.random.normal(loc = 0, scale = deltat)
                    #
                    diffusion_L0 = self.diffusion_L(state = ego_state_previous_moment, typee = diffusion_typee, OV_parameters = OV_paras)
                    diffusion_L = np.array([.0, two_dim_paras['sigma_long']*diffusion_L0, .0, .0, .0, .0])
                    #==========Jump
                    #jump_term is a float. 
                    jump_term = self.jump_term_L(state = ego_state_previous_moment, typee = jump_term_type, idm_paras = idm_paras, OV_parameters = OV_paras)
                    #jumpsize is a float. 
                    #print(len(jumpsizes_adapted_vehs_paths[veh_idx][path_idx]), time_idx)
                    jumpsize = jumpsizes_adapted_vehs_paths[veh_idx][path_idx][time_idx]
                    #
                    #new_state is a 1d array. 
                    #new_state,feasible_increment = self.TrimUpdate_single_dim(old_state = ego_state_previous_moment, deltat_sec = deltat, F = F, diffusion_L = diffusion_L, brownian = brownian, idm_paras = idm_paras)
                    if jerk_constraint:
                        #
                        old_delta_sec = delta_0
                        #old_old_state, ego_state_previous_previous_moment, the moment before the last moment. 
                        if jerk_constraint:
                            if t0 in sim_paths[veh_idx][path_idx].keys():
                                old_old_state = sim_paths[veh_idx][path_idx][t0]
                            else:
                                old_old_state = ego_state_previous_moment
                        #
                        new_state,feasible_increment = self.TrimUpdate_single_dim_with_jump(old_state = ego_state_previous_moment, deltat_sec = deltat, F = F, diffusion_L = diffusion_L, brownian = brownian, idm_paras = idm_paras, jumpsize = jump_term*jumpsize, jerk_constraint = jerk_constraint, old_old_state = old_old_state, old_delta_sec = old_delta_sec)
                        
                    else:
                        new_state,feasible_increment = self.TrimUpdate_single_dim_with_jump(old_state = ego_state_previous_moment, deltat_sec = deltat, F = F, diffusion_L = diffusion_L, brownian = brownian, idm_paras = idm_paras, jumpsize = jump_term*jumpsize)
                    #
                    sim_paths[veh_idx][path_idx][t2] = new_state
            #
        #########################################Convert the results to  the dataframe. 
        #simpaths_dict_of_pd[veh_idx][path_idx] = pd, columns are moments, which is the same as input arg ts. 
        simpaths_dict_of_pd = {}
        for veh_idx in range(len(init_states)):
            simpaths_dict_of_pd[veh_idx] = {pathidx:pd.DataFrame(sim_paths[veh_idx][pathidx])  for pathidx in range(N_paths)}
        #
        return simpaths_dict_of_pd
        pass
    
    
    
    @classmethod
    def sim_openroad_pure_HDVs_withjumps_twodim(self, adapted_ts_paths, \
            jumpsizes_adapted_paths_vehs_lon, jumpsizes_adapted_paths_vehs_lat, \
            init_states, leaderstates_paths, \
            tao_delay = .8, \
            idm_paras = idm_paras, \
            diffusion_typee = 'constant', \
            jump_term_type = 'constant', \
            acc_model = 'IDM', \
            OV_parameters = OV_paras, \
            jerk_constraint_lon = False, jerk_constraint_lat = False, \
            jump_or_not_lon = True, jump_or_not_lat = True, \
            #
            dim_paras = two_dim_paras, lw = 3.5,\
            stochastic_proecess_name = 'OU', \
            two_dim_paras = two_dim_paras, deltax_2d_or_not = False):
        """
        
        Simulation of the open road (non periodic).
        
        
        Difference:
        
            self.sim_openroad_pure_HDVs_withjumps
            self.sim_openroad_pure_HDVs_withjumps_twodim
            
        The latter is for two dim jump. 
        
        
            sim_paths = sim_openroad_pure_HDVs(ts, init_states, leader_states,  N_paths = 100, tao_delay = .8, diffusion_L = diffusion_L)
        
        --------------------------------------------------------------------
        @input: jump_term_type
        
            may be False, 'constant','parabolic','OV_Ngoduy','speed_dependent_jerk'
            
        @input: adapted_ts_paths
        
            adapted_ts = adapted_ts_paths[path_idx]
        
            adapted_ts is a list contaiing the lists. 
            
            jumpsizes_adapted = jumpsizes_adapted_vehs_paths[veh_idx][path_idx],is either 0 or a jumptize. 
            
            Generation methods:
            
            ###############Get adapted_ts, jump_or_nots_lists,jumpsizes_adapted_lists###############################################
            reload(irre)
            T_horizon_sec = 1000
            #
            #Number of vehicles that is irregular. 
            N_irregulars = 5
            #
            #distributions of the jump sizes. 
            args_jumpsize_diss = [{'mu':.0, 'sigma': 1.0} for i in range(N_irregulars)]
            jumpdensities =  [2 for i in range(N_irregulars)]
            #
            #
            jumptimes_es,jumpsizes_es = irre.JumpDiffusionDrivingBehavior.JumpMomentsWithJumpSizes_multiplevehicle(T_horizon_sec= T_horizon_sec, args_jumpsize_diss = args_jumpsize_diss, jumpdensities = jumpdensities)
            #
            #determine the ts (a grid) and the adapted_ts
            ts = np.linspace(0, T_horizon_sec, int(T_horizon_sec*2))
            adapted_ts,jump_or_nots_lists,jumpsizes_adapted_lists = irre.JumpDiffusionDrivingBehavior.AdaptEuler(ts = ts, \
                                            jumptimes_lists = jumptimes_es, jumpsizes_lists = jumpsizes_es, )
            
            ##############################################################
        
            
        @input: init_states
        
            a list of states containing the intiail state of the vehicles, thery are the 1st folloer, second follower...
            
            state = init_states[veh_idx], state = [x, vx, y, vy, zlon, zlat]
            
            
            len(jumpsizes_adapted_paths_vehs_lon[path_idx])==len(init_states)
            
        @input: jumpsizes_adapted_paths_vehs_lon, jumpsizes_adapted_paths_vehs_lat
        
            both are dict. 
            
            jumpsizes_adapted_paths_vehs_lon[path_idx][veh_idx] = [size1, size2 ,..]
            
            The length should be the same as adapted_ts.
        
        @input: jerk_constraint
        
            a bool. Used to indicate that whether the max and min jerk constraint is applied. 
            
                influence the function, extra input args:
                
                    self.TrimUpdate(jerk_constraint = False, old_old_state = False, old_delta_sec = False)
                
            
                minn,maxx = TwoDimMicroModel.Speed_upperlower_given_jerk_upperlower(v0, v1, deltat0, delta1, )
        
        @input: acc_model
        
            the acceleration model used. 
            
            It can be: 'IDM', 'OV_Ngoduy', 'OV_peng'
            
            OV_peng: 
            
        @input: diffusion_typee = 'constant'
        
            used in self.diffusion_L
            
            can be 'constant', 'parabolic', 'lognormal', 'speed_dependent_jerk', 'speed_dependent_jerk_Bouadi'
        @input: adapted_ts_paths
            
            adapted_ts_paths[path_idx] = [t0, t1, t2,...]
        
        @input: diffusion_L
        
            a function that represent the diffusion term in the model
        
        @input: leaderstates_paths
        
            leader_states = leaderstates_paths[path_idx]
        
            an dict of 1darray.
             
            leader_states[time] = veh_state, each state is [x, vx, y, vy, zlon, zlat]
            
            It can be obtained via:
                
                #-------------
                leaderstates = irre.SingleLaneSim.LeaderVehicleState_constancespeed(ts = adapted_ts, v_ms = 40)
        
        @input: vehs_types
        
            a list of the vehicles types. 
        
        @OUTPUT: sim_paths
            
            sim_paths[veh_idx][path_idx] = pd, columns are moments, which is the same as input arg ts.
            
            veh_paths = sim_path[idx_veh] is a dict. 
            veh_path = veh_paths[path_idx]
            
                is a pd. Columns are moments, which is consistemt with input ts. 
        
        """
        #==========Check the moment is correct. 
        if len(adapted_ts_paths)!=len(leaderstates_paths):
            raise ValueError('sdsdfsdfsadf')
        #========Check N_paths
        N_paths = len(jumpsizes_adapted_paths_vehs_lon)
        
        #
        #sim_paths[path_idx][veh_idx][moment] = state, state = [x, vx,y, vy, zlon, zlat]
        sim_paths = {pathidx:{veh_idx:{min(adapted_ts_paths[pathidx]):init_states[veh_idx]} for veh_idx in range(len(init_states))} for pathidx in range(N_paths)}
        #
        for path_idx in range(N_paths):
            #
            adapted_ts = adapted_ts_paths[path_idx]
            #
            #for time_idx,(t0,t1) in enumerate(zip(adapted_ts[:-1], adapted_ts[1:])):
            #   tmp is used to iterate the time. 
            tmp = [min(adapted_ts)-np.diff(adapted_ts)[0]] + list(adapted_ts[:-2])
            for time_idx,(t0,t1,t2) in enumerate(zip(tmp, adapted_ts[:-1], adapted_ts[1:])):
                #
                delta_0 = t1 - t0
                deltat_sec = t2 - t1
                #
                for veh_idx in range(len(init_states)):
                    #=========FIND ego state and the leader state at certain moment earlier. 
                    #print(time_idx, t1)
                    old_delta_sec = t1 - t0
                    if t0 not in sim_paths[path_idx][veh_idx].keys():
                        ego_state_previous_previous = sim_paths[path_idx][veh_idx][t1]
                    else:
                        ego_state_previous_previous = sim_paths[path_idx][veh_idx][t0]
                    #
                    ego_state_previous_moment = sim_paths[path_idx][veh_idx][t1]
                    #
                    x,vx,y,vy,zlon,zlat = ego_state_previous_moment
                    #------------find its leader sate in  leader_state
                    t_leader = t1 - tao_delay
                    #
                    if t_leader<=min(adapted_ts):
                        if veh_idx==0:
                            leader_state = leaderstates_paths[path_idx][min(adapted_ts)]
                        else:
                            leader_state = sim_paths[path_idx][veh_idx - 1][min(adapted_ts)]
                    else:
                        #
                        tmp = np.abs(t_leader - np.array(adapted_ts))
                        t_key = np.where(tmp==min(tmp))[0][0]
                        #
                        if veh_idx==0:
                            leader_state = leaderstates_paths[path_idx][min(adapted_ts)]
                        else:
                            leader_state = sim_paths[path_idx][veh_idx - 1][min(adapted_ts)]
                    #=========Calculate the deltax
                    #
                    #F = np.array([vx, acceleration, .0, .0, .0, .0])
                    F,potentials_dict = TwoDimMicroModel.F_SingleLane(\
                        ego_state  = ego_state_previous_moment, \
                        ego_lane_lw = lw, \
                        ego_lane_middle_line_coor = .0, \
                        ego_lane_marks_coor = (-lw/2.0, lw/2.0), \
                        road_bounds = (-lw/2.0, lw/2.0), \
                        lw_boundary_lanes = (lw, lw), \
                        leader_state = leader_state, \
                        two_dim_paras = two_dim_paras, \
                        stochastic_proecess_name = stochastic_proecess_name, \
                        deltax_2d_or_not = deltax_2d_or_not)
                    diffusion_L = TwoDimMicroModel.L(state  = ego_state_previous_moment, two_dim_paras = two_dim_paras, stochastic_proecess_name = stochastic_proecess_name)
                    brownian = np.random.normal(loc = np.zeros((2,)), scale = deltat_sec)
                    #brownian = np.random.normal(loc = 0, scale = deltat_sec)
                    #==========Jump
                    #jump_term is a float. 
                    jump_term = self.jump_term_L(state = ego_state_previous_moment, typee = jump_term_type, idm_paras = idm_paras, OV_parameters = OV_paras)
                    #
                    jumpsize_lon = jumpsizes_adapted_paths_vehs_lon[path_idx][veh_idx][time_idx]
                    jumpsize_lat = jumpsizes_adapted_paths_vehs_lat[path_idx][veh_idx][time_idx]
                    #
                    new_state,feasible_increment = self.TrimUpdate_two_dim_jump(\
                        old_state = ego_state_previous_moment, \
                        deltat_sec = deltat_sec, \
                        F = F, L = diffusion_L, \
                        brownian = brownian, \
                        idm_paras = idm_paras, \
                        jumpsize_lon = jumpsize_lon, \
                        jumpsize_lat = jumpsize_lat,\
                        jump_or_not_lon = jump_or_not_lon, \
                        jump_or_not_lat = jump_or_not_lat, \
                        jerk_constraint_lon = jerk_constraint_lon, \
                        jerk_constraint_lat = jerk_constraint_lat, \
                        old_old_state = ego_state_previous_previous, \
                        old_delta_sec = old_delta_sec )
                    #
                    sim_paths[path_idx][veh_idx][t2] = new_state
            #
        #########################################Convert the results to  the dataframe. 
        #simpaths_dict_of_pd[path_idx][veh_idx], columns are moments, which is the same as input arg ts. 
        simpaths_dict_of_pd = {}
        for path_idx in sim_paths.keys():
            simpaths_dict_of_pd[path_idx] = {veh_idx:pd.DataFrame(sim_paths[path_idx][veh_idx]) for veh_idx in sim_paths[path_idx].keys()}
        #
        return simpaths_dict_of_pd
        
    
    
    @classmethod
    def sim_openroad_pure_HDVs(self, ts, init_states, leader_states, N_paths = 100, tao_delay = .8, idm_paras = idm_paras, diffusion_typee = 'constant', acc_model = 'IDM', OV_parameters = OV_paras):
        """
        Simulation of the open road (non periodic).
        
            sim_paths = sim_openroad_pure_HDVs(ts, init_states, leader_states,  N_paths = 100, tao_delay = .8, diffusion_L = diffusion_L)
        
        --------------------------------------------------------------------
        @input: acc_model
        
            the acceleration model used. 
            
            It can be: 'IDM', 'OV_Ngoduy', 'OV_peng'
            
        @input: diffusion_typee = 'constant'
        
            used in self.diffusion_L
            
            can be 'constant', 'parabolic', 'lognormal', 'speed_dependent_jerk'
            
        @input: init_states
        
            a list of states containing the intiail state of the vehicles, thery are the 1st folloer, second follower...
            
            state = init_states[veh_idx], state = [x, vx, y, vy, zlon, zlat]
        
        @input: jumpsizes_adapted
        
            jumpsizes_adapted = jumpsizes_adapted_lists[idx], 
            
            jumpsizes_adapted (a list of float. ) containes the jump size for irregular vehicle idx. 
            
        @input: diffusion_L
        
            a function that represent the diffusion term in the model
        
        @input: leader_states
        
            an dict of 1darray.
             
            leader_states[time] = veh_state, each state is [x, vx, y, vy, zlon, zlat]
        
        @input: vehs_types
        
            a list of the vehicles types. 
        
        @OUTPUT: sim_paths
        
            
            veh_paths = sim_path[idx_veh] is a dict. 
            veh_path = veh_paths[path_idx]
            
                is a pd. Columns are moments, which is consistemt with input ts. 
        
        """
        
        
        #sim_paths[veh_idx][path_idx][moment] = state, state = [x, vx,y, vy, zlon, zlat]
        sim_paths = {veh_idx:{pathidx:{min(ts):init_states[veh_idx]} for pathidx in range(N_paths)} for veh_idx in range(len(init_states))}
        #
        for path_idx in range(N_paths):
            for t0,t1 in zip(ts[:-1], ts[1:]):
                #
                deltat = t1 - t0
                #
                for veh_idx in range(len(init_states)):
                    #=========FIND ego state and the leader state at certain moment earlier. 
                    ego_state_previous_moment = sim_paths[veh_idx][path_idx][t0]
                    x,vx,y,vy,zlon,zlat = ego_state_previous_moment
                    #------------find its leader
                    t_leader = t1 - tao_delay
                    #
                    if t_leader<=min(ts):
                        if veh_idx==0:
                            leader_state = leader_states[min(ts)]
                        else:
                            leader_state = sim_paths[veh_idx - 1][path_idx][min(ts)]
                    else:
                        tmp = np.abs(t_leader - np.array(ts))
                        t_key = np.where(tmp==min(tmp))[0][0]
                        if veh_idx==0:
                            leader_state = leader_states[min(ts)]
                        else:
                            leader_state = sim_paths[veh_idx - 1][path_idx][min(ts)]
                    #=========Calculate the deltax
                    deltax = leader_state[0] - ego_state_previous_moment[0]
                    #
                    #========Aceleration,
                    if acc_model=='IDM':
                        acceleration  = TwoDimMicroModel.Potential_IDM(v_self = ego_state_previous_moment[1]*3.6, v_leader = leader_state[1]*3.6, deltax =deltax, idm_paras = idm_paras)
                    elif acc_model=='OV_Ngoduy':
                        acceleration  = TwoDimMicroModel.Potential_OV_Ngoduy(v_self = ego_state_previous_moment[1]*3.6, v_leader = leader_state[1]*3.6, deltax =deltax, OV_paras = OV_paras)
                    elif acc_model=='OV_peng':
                        acceleration  = TwoDimMicroModel.Potential_OV_Peng(v_self = ego_state_previous_moment[1]*3.6, v_leader = leader_state[1]*3.6, deltax =deltax, OV_paras = OV_paras)
                        
                    F = np.array([vx, acceleration, .0, .0, .0, .0])
                    #
                    #brownian = np.random.normal(loc = np.zeros((2,)), scale = deltat_sec)
                    brownian = np.random.normal(loc = 0, scale = deltat)
                    diffusion_L0 = self.diffusion_L(state = ego_state_previous_moment, typee = diffusion_typee, OV_parameters = OV_paras)
                    #
                    diffusion_L = np.array([.0, diffusion_L0, .0, .0, .0, .0])
                    #new_state is a 1d array. 
                    new_state,feasible_increment = self.TrimUpdate_single_dim(old_state = ego_state_previous_moment, deltat_sec = deltat, F = F, diffusion_L = diffusion_L, brownian = brownian, idm_paras = idm_paras)
                    #
                    sim_paths[veh_idx][path_idx][t1] = new_state
            #
        #simpaths_dict_of_pd[veh_idx][path_idx] = pd, columns are moments, which is the same as input arg ts. 
        simpaths_dict_of_pd = {}
        for veh_idx in range(len(init_states)):
            simpaths_dict_of_pd[veh_idx] = {pathidx:pd.DataFrame(sim_paths[veh_idx][pathidx])  for pathidx in range(N_paths)}
        #
        return simpaths_dict_of_pd
        pass
        

    def sim_insert_all_vehs_at_once_jump(self, adapted_ts, jumpsizes_adapted_lists, \
            desired_density = 100, \
            inserted_initial_state = np.array([.0, .0, .0, 0, .000001, .00001]), safety_gap2downstream_when_insert = 10, \
            safety_gap2upstream_when_insert = 10,  \
            idm_paras = idm_paras, \
            stochastic_proecess_name = 'OU', \
            two_dim_paras = two_dim_paras, \
            leader_state_infinity = np.array([1e10, .0, .0, 0, .0, .0]), \
            intert_tolerance = 1e-1, \
            deltax_2d_or_not = False, \
            idm_paras_irregular = idm_paras, \
            two_dim_paras_irregular = two_dim_paras, \
            jump_or_not = True, delay_tao_sec = .0, veh_length_m = 5, jerk_constraint = False,):
        """
        Number of vehicles that is irregular is given by len(jumpsizes_adapted_lists)
        
        jumpsizes_adapted = jumpsizes_adapted_lists[idx], 
        
        jumpsizes_adapted (a list of float. ) containes the jump size for irregular vehicle idx. 
        
        
        ------------------------------------------------------------
        @input: jerk_constraint
        
            a bool. Used to indicate that whether the max and min jerk constraint is applied. 
            
                influence the function, extra input args:
                
                    self.TrimUpdate(jerk_constraint = False, old_old_state = False, old_delta_sec = False)
                
            
                minn,maxx = TwoDimMicroModel.Speed_upperlower_given_jerk_upperlower(v0, v1, deltat0, delta1, )
        
        
        
        @input: adapted_ts, jumpsizes_adapted_lists
        
            Generation methods:
            
            ##############################################################
            reload(irre)
            T_horizon_sec = 1000
            #
            #Number of vehicles that is irregular. 
            N_irregulars = 5
            #
            #distributions of the jump sizes. 
            args_jumpsize_diss = [{'mu':.0, 'sigma': 1.0} for i in range(N_irregulars)]
            jumpdensities =  [2 for i in range(N_irregulars)]
            #
            #
            jumptimes_es,jumpsizes_es = irre.JumpDiffusionDrivingBehavior.JumpMomentsWithJumpSizes_multiplevehicle(T_horizon_sec= T_horizon_sec, args_jumpsize_diss = args_jumpsize_diss, jumpdensities = jumpdensities)
            #
            #determine the ts (a grid) and the adapted_ts
            ts = np.linspace(0, T_horizon_sec, int(T_horizon_sec*2))
            adapted_ts,jump_or_nots_lists,jumpsizes_adapted_lists = irre.JumpDiffusionDrivingBehavior.AdaptEuler(ts = ts, \
                                            jumptimes_lists = jumptimes_es, jumpsizes_lists = jumpsizes_es, )
            
            ##############################################################
        
        @input: delay_tao_sec
        
            the delay of the vehicle. 
        
        @input: jump_or_not
        
            whether consider the jump term or not. It will be used in the TrimUpdate() method. 
            
        @input: consider_jump
        
            whether consider the jump. 
            
        @input: idm_paras_irregular and two_dim_paras_irregular
        
            the parameters set of the irregular parameters. 
        
        @input: N_irregular_vehs
        
            the number of irregular vehicles. 
            
            len(jumpsizes_adapted_lists) = N_irregular_vehs
            
        @OUPUT: STATEs, irregular_vehs_ids
            
            STATES[T] = copy.deepcopy(self.vehs_dict)
            self.vehs_dict[vid] = new_state
            
            irregular_vehs_ids is a list containing the ids of the irregular vehicls. 
            
            
        """
        #returned value
        STATES = {min(adapted_ts):{}}
        
        #
        N_irregular_vehs = len(jumpsizes_adapted_lists)
        #a list containing the irregular vehicles. 
        #   len()
        irregular_vehs_ids = []
        #jumpsizes_vehs[vid] = jumpsizes_adapted
        jumpsizes_vehs = {}
        
        #======================Insert vehicles. 
        vehs_number = int(max(2, min(desired_density/1000.0*(self.length-1), (self.length-1)/1000.0*135)))
        space = (self.length-1.0)/vehs_number
        equilibrium_v = TwoDimMicroModel.IDM_equilibrium_v_from_deltax(deltax = space)
        #   calculate the locations for insertin the vehicles. locs_inserted
        locs_inserted = [0.5]
        for i in range(vehs_number):
            tmp = locs_inserted[-1] + space
            if tmp<=self.length:
                locs_inserted.append(tmp)
            else:
                break
        #----------Initi the vehicles
        n_irregular_veh_assigned = 0
        irre_idx_selected = np.random.choice(range(len(locs_inserted)), min(len(locs_inserted), N_irregular_vehs))
        for i,loc in enumerate(locs_inserted):
            #   record the irregular veh id and the jumpsizes. 
            #   note that -1 is bevause the index is counted from 0. 
            #if i<=N_irregular_vehs - 1:
            if i in irre_idx_selected:
                irregular_vehs_ids.append(i)
                tmp_idx = len(jumpsizes_vehs)
                #jumpsizes_vehs keys are the vids. 
                #print()
                jumpsizes_vehs[i] = jumpsizes_adapted_lists[tmp_idx]
            #
            state = np.array([loc, equilibrium_v, self.lw/2.0, 1e-5, .0000001, .0000001])#inserted_initial_state
            #state[0] = loc
            #state[2] = self.lw/2.0
            #np.array([loc, .0, self.lw/2.0, .0, .0, .0])
            self.vehs_at_lanes.append(i)
            #self.vehs_dict[vehicle_id] = [x, vx, y, vy, zlon, zlat]
            self.vehs_dict[i] = state
        STATES[min(adapted_ts)] = copy.deepcopy(self.vehs_dict)
            
        #==============================SIMULATION ITERATION
        #
        T = 0
        #while T<T_horizon_sec:
        for idx_time,deltat_sec in enumerate(np.diff(adapted_ts)):
            T = adapted_ts[idx_time + 1]
            self.snapshots[T] = {}
            #=======================================================
            #calcualte F and L. keys are the vehicle ids. 
            #   F_dicts[vid] is a 1d array. 
            #   L_dicts[vid] is a N*2 array. 2 means lon and lat noise. 
            F_dicts = {}
            L_dicts = {}
            leader_state_dict = {}
            potentials_dict = {}
            
            #=======================================================Calculate the F and L
            for i,vid in enumerate(self.vehs_at_lanes):
                #===================get ego state 
                ego_state = self.vehs_dict[vid]
                #---get ego state at previous moment and delta_sec on its previous moment
                #   self.TrimUpdate(jerk_constraint = False, old_old_state = False, old_delta_sec = False)
                if idx_time<=1:
                    old_delta_sec = adapted_ts[1]-adapted_ts[0]
                    ego_state_previous = copy.deepcopy(ego_state)
                    #if ego_state_previous[0]
                else:
                    #
                    old_delta_sec = adapted_ts[idx_time]-adapted_ts[idx_time-1]
                    #print(self.snapshots.keys())
                    ego_state_previous = self.snapshots[adapted_ts[idx_time]][vid]['S']
                #
                #===================Leader state
                #-------------------Determine the time minus the delay_tao_sec (stored in t_considering_delay)
                tmp_t = T - delay_tao_sec
                if tmp_t<min(STATES.keys()):
                    t_considering_delay = min(STATES.keys())
                else:
                    tmp_delta =  abs(np.array(sorted(STATES.keys()))- tmp_t)
                    idx_tmp = np.where(tmp_delta == min(tmp_delta))[0][0]
                    t_considering_delay = np.array(sorted(STATES.keys()))[idx_tmp]
                #-------------------find the leader 
                if len(self.vehs_at_lanes)==1:
                    #if only one vehicle
                    leader_state = copy.deepcopy(ego_state)
                    leader_state[0] = ego_state[0]+1e10
                else:
                    #if vid is the most downsteeam vehicle (in the periodic condition)
                    if i==len(self.vehs_at_lanes)-1:
                        #
                        leader_id = self.vehs_at_lanes[0]
                        #leader_state = copy.deepcopy(self.vehs_dict[leader_id])
                        leader_state = STATES[t_considering_delay][leader_id]
                        leader_state[0] = leader_state[0]+ self.length
                        
                    else:
                        leader_id = self.vehs_at_lanes[i+1]
                        #leader_state = self.vehs_dict[leader_id]
                        leader_state = STATES[t_considering_delay][leader_id]
                    #------------------search leader at ego lane.
                leader_state_dict[vid] = copy.deepcopy(leader_state)
                #
                #-----------------
                #--------------calculate the F and L
                #   F_dicts[vid] is a 1d array. 
                F_dicts[vid],potentials_dict[vid] = TwoDimMicroModel.F_SingleLane(ego_state  = ego_state, \
                    ego_lane_lw = self.lw, \
                    ego_lane_middle_line_coor = self.lw/2.0, \
                    ego_lane_marks_coor = (0, self.lw), \
                    road_bounds = (0, self.lw), \
                    lw_boundary_lanes = (self.lw, self.lw), \
                    leader_state = leader_state, \
                    two_dim_paras = two_dim_paras, \
                    stochastic_proecess_name = stochastic_proecess_name, deltax_2d_or_not = deltax_2d_or_not)
                L_dicts[vid] = TwoDimMicroModel.L(state  = ego_state, two_dim_paras = two_dim_paras, stochastic_proecess_name = stochastic_proecess_name)
                
            ################################UPDATE, jump is in this stage.
            for vid in self.vehs_dict.keys():
                #
                #jumpsizes_vehs[veh_idx][path_idx]
                if vid in jumpsizes_vehs.keys():
                    jumpsize = jumpsizes_vehs[vid][0][idx_time + 1]
                else:
                    jumpsize = .0
                #
                brownian = np.random.normal(loc = np.zeros((2,)), scale = deltat_sec)
                #
                #print(self.vehs_dict[vid], vid in F_dicts, vid in L_dicts)
                #
                #new_state = self.vehs_dict[vid] + F_dicts[vid] + np.matmul(L_dicts[vid], brownian/deltat_sec)
                #self.TrimUpdate(jerk_constraint = False, old_old_state = False, old_delta_sec = False)
                if jerk_constraint:
                    new_state,feasible_increment = self.TrimUpdate(old_state = self.vehs_dict[vid], deltat_sec = deltat_sec, F = F_dicts[vid], L = L_dicts[vid], brownian = brownian, idm_paras = idm_paras, jumpsize = jumpsize, jump_or_not = jump_or_not, old_old_state = ego_state_previous, old_delta_sec = old_delta_sec, jerk_constraint = jerk_constraint)
                else:
                    new_state,feasible_increment = self.TrimUpdate(old_state = self.vehs_dict[vid], deltat_sec = deltat_sec, F = F_dicts[vid], L = L_dicts[vid], brownian = brownian, idm_paras = idm_paras, jumpsize = jumpsize, jump_or_not = jump_or_not)
                #
                #print(F_dicts[vid].shape, np.matmul(L_dicts[vid], brownian).shape)
                #new_state = self.vehs_dict[vid] + F_dicts[vid]*deltat_sec + np.matmul(L_dicts[vid], brownian)
                #---------------------UPDATE systen variables. 
                #SNAPSHOT
                self.snapshots[T]['vehs_at_lanes'] = copy.deepcopy(self.vehs_at_lanes)
                self.snapshots[T][vid] = {'S':copy.deepcopy(self.vehs_dict[vid]), \
                    'feasibleincrement':copy.deepcopy(feasible_increment), \
                    'L':copy.deepcopy(L_dicts[vid]), \
                    'F':copy.deepcopy(F_dicts[vid]), \
                    'potentials':copy.deepcopy(potentials_dict[vid]), }

                #
                #---------------------------------Update the vehicle order. 
                if new_state[0]>self.length:
                    new_state[0] = new_state[0] - self.length
                    #print(new_state.shape)
                    self.vehs_dict[vid] = new_state
                    #
                    tmp = copy.deepcopy([self.vehs_at_lanes[-1]] + self.vehs_at_lanes[:-1])
                    self.vehs_at_lanes = tmp
                else:
                    #print(new_state.shape)
                    self.vehs_dict[vid] = new_state
            #
            #############################RECORD the state
            STATES[T] = copy.deepcopy(self.vehs_dict)
            
            #
            builtins.tmp = STATES
        
        return STATES,irregular_vehs_ids

    def sim_insert_all_vehs_at_once_two_dim_jump(self, adapted_ts, \
            #longitudinal and lateral jump information
            #jumpsizes_adapted_lists_lon[vid][path_idx]
            jumpsizes_adapted_lists_lon, \
            jumpsizes_adapted_lists_lat, \
            jump_or_not_lon = True, jump_or_not_lat = True, \
            #
            desired_density = 100, \
            inserted_initial_state = np.array([.0, .0, .0, 0, .000001, .00001]), safety_gap2downstream_when_insert = 10, \
            safety_gap2upstream_when_insert = 10,  \
            idm_paras = idm_paras, \
            stochastic_proecess_name = 'OU', \
            two_dim_paras = two_dim_paras, \
            leader_state_infinity = np.array([1e10, .0, .0, 0, .0, .0]), \
            intert_tolerance = 1e-1, \
            deltax_2d_or_not = False, \
            idm_paras_irregular = idm_paras, \
            two_dim_paras_irregular = two_dim_paras, \
            delay_tao_sec = .0, veh_length_m = 5, \
            jerk_constraint_lon = False, jerk_constraint_lat = False, start_from_speed = .0):
        """
        Number of vehicles that is irregular is given by len(jumpsizes_adapted_lists)
        
        jumpsizes_adapted = jumpsizes_adapted_lists[idx], 
        
        jumpsizes_adapted (a list of float. ) containes the jump size for irregular vehicle idx. 
        
        Difference:
        
            self.sim_insert_all_vehs_at_once_jump
            self.sim_insert_all_vehs_at_once_two_dim_jump
            
        The latter is for the 
        
        
        ------------------------------------------------------------
        @input: start_from_zero_speed
        
            the longitudinal speed is zero or not at initial state. 
            
        @input: jerk_constraint
        
            a bool. Used to indicate that whether the max and min jerk constraint is applied. 
            
                influence the function, extra input args:
                
                    self.TrimUpdate(jerk_constraint = False, old_old_state = False, old_delta_sec = False)
                
            
                minn,maxx = TwoDimMicroModel.Speed_upperlower_given_jerk_upperlower(v0, v1, deltat0, delta1, )
        
        
        
        @input: adapted_ts_lon, jumpsizes_adapted_lists_lon
        
            Generation methods:
            
            ##############################################################
            reload(irre)
            T_horizon_sec = 1000
            #
            #Number of vehicles that is irregular. 
            N_irregulars = 5
            #
            #distributions of the jump sizes. 
            args_jumpsize_diss = [{'mu':.0, 'sigma': 1.0} for i in range(N_irregulars)]
            jumpdensities =  [2 for i in range(N_irregulars)]
            #
            #
            jumptimes_es,jumpsizes_es = irre.JumpDiffusionDrivingBehavior.JumpMomentsWithJumpSizes_multiplevehicle(T_horizon_sec= T_horizon_sec, args_jumpsize_diss = args_jumpsize_diss, jumpdensities = jumpdensities)
            #
            #determine the ts (a grid) and the adapted_ts
            ts = np.linspace(0, T_horizon_sec, int(T_horizon_sec*2))
            adapted_ts_lon,jump_or_nots_lists_lon,jumpsizes_adapted_lists_lon = irre.JumpDiffusionDrivingBehavior.AdaptEuler(ts = ts, \
                                            jumptimes_lists = jumptimes_es, jumpsizes_lists = jumpsizes_es, )
            
            ##############################################################
        
        @input: delay_tao_sec
        
            the delay of the vehicle. 
        
        @input: jump_or_not
        
            whether consider the jump term or not. It will be used in the TrimUpdate() method. 
            
        @input: consider_jump
        
            whether consider the jump. 
            
        @input: idm_paras_irregular and two_dim_paras_irregular
        
            the parameters set of the irregular parameters. 
        
        @input: N_irregular_vehs
        
            the number of irregular vehicles. 
            
            len(jumpsizes_adapted_lists) = N_irregular_vehs
            
        @OUPUT: STATEs, irregular_vehs_ids
            
            STATES[T] = copy.deepcopy(self.vehs_dict)
            self.vehs_dict[vid] = new_state
            
            irregular_vehs_ids is a list containing the ids of the irregular vehicls. 
            
            
        """
        #returned value
        STATES = {min(adapted_ts):{}}
        #
        N_irregular_vehs = len(jumpsizes_adapted_lists_lon)
        #a list containing the irregular vehicles. 
        #   len()
        irregular_vehs_ids = []
        #jumpsizes_vehs[vid] = jumpsizes_adapted
        jumpsizes_vehs = {}
        
        #======================Insert vehicles. 
        vehs_number = int(max(1, min(desired_density/1000.0*(self.length-1), (self.length-1)/1000.0*135)))
        #
        space = (self.length-1.0)/vehs_number
        equilibrium_v = TwoDimMicroModel.IDM_equilibrium_v_from_deltax(deltax = space)
        #   calculate the locations for insertin the vehicles. locs_inserted
        locs_inserted = []
        for i in range(vehs_number):
            if len(locs_inserted)==0:
                locs_inserted.append(0.5)
            else:
                tmp = locs_inserted[-1] + space
                if tmp<=self.length:
                    locs_inserted.append(tmp)
                else:
                    break
        #===============================Initiate the vehicles
        #First determine the irregular vehicles, the id is stored in irre_idx_selected, a iterable.
        #the info of jumpsize of the irregular vehicles is stored in jumpsizes_vehs_lon and jumpsizes_vehs_lat.
        #jumpsizes_vehs[vid_ireegular] = 
        n_irregular_veh_assigned = 0
        irre_idx_selected = np.random.choice(range(len(locs_inserted)), min(len(locs_inserted), N_irregular_vehs))
        #
        jumpsizes_vehs_lon = {}
        jumpsizes_vehs_lat = {}
        for i,loc in enumerate(locs_inserted):
            #   record the irregular veh id and the jumpsizes. 
            #   note that -1 is bevause the index is counted from 0. 
            #if i<=N_irregular_vehs - 1:
            if i in irre_idx_selected:
                irregular_vehs_ids.append(i)
                tmp_idx = len(jumpsizes_vehs)
                #jumpsizes_vehs keys are the vids. 
                #print()
                jumpsizes_vehs_lon[i] = jumpsizes_adapted_lists_lon[tmp_idx]
                jumpsizes_vehs_lat[i] = jumpsizes_adapted_lists_lat[tmp_idx]
            #
            if isinstance(start_from_speed, bool):
                state = np.array([loc, equilibrium_v, self.lw/2.0, 1e-5, .0000001, .0000001])
            else:
                state = np.array([loc, start_from_speed, self.lw/2.0, 1e-5, .0000001, .0000001])#inserted_initial_state
                #state = np.array([loc, equilibrium_v, self.lw/2.0, 1e-5, .0000001, .0000001])#inserted_initial_state
            #state[0] = loc
            #state[2] = self.lw/2.0
            #np.array([loc, .0, self.lw/2.0, .0, .0, .0])
            self.vehs_at_lanes.append(i)
            #self.vehs_dict[vehicle_id] = [x, vx, y, vy, zlon, zlat]
            self.vehs_dict[i] = state
        STATES[min(adapted_ts)] = copy.deepcopy(self.vehs_dict)
            
        #==============================SIMULATION ITERATION
        #
        T = 0
        #while T<T_horizon_sec:
        for idx_time,deltat_sec in enumerate(np.diff(adapted_ts)):
            T = adapted_ts[idx_time + 1]
            self.snapshots[T] = {}
            #=======================================================
            #calcualte F and L. keys are the vehicle ids. 
            #   F_dicts[vid] is a 1d array. 
            #   L_dicts[vid] is a N*2 array. 2 means lon and lat noise. 
            F_dicts = {}
            L_dicts = {}
            leader_state_dict = {}
            potentials_dict = {}
            #=======================================================Calculate the F and L
            for i,vid in enumerate(self.vehs_at_lanes):
                #===================get ego state 
                ego_state = self.vehs_dict[vid]
                #---get ego state at previous moment and delta_sec on its previous moment
                #   self.TrimUpdate(jerk_constraint = False, old_old_state = False, old_delta_sec = False)
                if idx_time<=1:
                    old_delta_sec = adapted_ts[1]-adapted_ts[0]
                    ego_state_previous = copy.deepcopy(ego_state)
                    #if ego_state_previous[0]
                else:
                    #
                    old_delta_sec = adapted_ts[idx_time]-adapted_ts[idx_time-1]
                    #print(self.snapshots.keys())
                    ego_state_previous = self.snapshots[adapted_ts[idx_time]][vid]['S']
                #
                #===================Leader state
                #-------------------Determine the time minus the delay_tao_sec (stored in t_considering_delay)
                tmp_t = T - delay_tao_sec
                if tmp_t<min(STATES.keys()):
                    t_considering_delay = min(STATES.keys())
                else:
                    tmp_delta =  abs(np.array(sorted(STATES.keys()))- tmp_t)
                    idx_tmp = np.where(tmp_delta == min(tmp_delta))[0][0]
                    t_considering_delay = np.array(sorted(STATES.keys()))[idx_tmp]
                #-------------------find the leader 
                if len(self.vehs_at_lanes)==1:
                    #if only one vehicle
                    leader_state = copy.deepcopy(ego_state)
                    leader_state[0] = ego_state[0]+1e10
                else:
                    #if vid is the most downsteeam vehicle (in the periodic condition)
                    if i==len(self.vehs_at_lanes)-1:
                        #
                        leader_id = self.vehs_at_lanes[0]
                        #leader_state = copy.deepcopy(self.vehs_dict[leader_id])
                        leader_state = STATES[t_considering_delay][leader_id]
                        leader_state[0] = leader_state[0]+ self.length
                        
                    else:
                        leader_id = self.vehs_at_lanes[i+1]
                        #leader_state = self.vehs_dict[leader_id]
                        leader_state = STATES[t_considering_delay][leader_id]
                    #------------------search leader at ego lane.
                leader_state_dict[vid] = copy.deepcopy(leader_state)
                #
                #-----------------
                #--------------calculate the F and L
                #   F_dicts[vid] is a 1d array. 
                F_dicts[vid],potentials_dict[vid] = TwoDimMicroModel.F_SingleLane(ego_state  = ego_state, \
                    ego_lane_lw = self.lw, \
                    ego_lane_middle_line_coor = self.lw/2.0, \
                    ego_lane_marks_coor = (0, self.lw), \
                    road_bounds = (0, self.lw), \
                    lw_boundary_lanes = (self.lw, self.lw), \
                    leader_state = leader_state, \
                    two_dim_paras = two_dim_paras, \
                    stochastic_proecess_name = stochastic_proecess_name, \
                    deltax_2d_or_not = deltax_2d_or_not)
                L_dicts[vid] = TwoDimMicroModel.L(state  = ego_state, two_dim_paras = two_dim_paras, stochastic_proecess_name = stochastic_proecess_name)
                
            ################################UPDATE, jump is in this stage.
            for vid in self.vehs_dict.keys():
                #
                #jumpsizes_vehs_lon[veh_idx][path_idx]
                if vid in jumpsizes_vehs_lon.keys():
                    #note that the index 0 is for path_idx, useless in this method. 
                    jumpsize_lon = jumpsizes_vehs_lon[vid][0][idx_time + 1]
                    jumpsize_lat = jumpsizes_vehs_lat[vid][0][idx_time + 1]
                else:
                    jumpsize_lon = .0
                    jumpsize_lat = .0
                #
                brownian = np.random.normal(loc = np.zeros((2,)), scale = deltat_sec)
                #
                new_state,feasible_increment = self.TrimUpdate_two_dim_jump(\
                    old_state = self.vehs_dict[vid], \
                    deltat_sec = deltat_sec, \
                    F = F_dicts[vid], L = L_dicts[vid], \
                    brownian = brownian, \
                    idm_paras = idm_paras, \
                    jumpsize_lon = jumpsize_lon, \
                    jumpsize_lat = jumpsize_lat,\
                    jump_or_not_lon = jump_or_not_lon, \
                    jump_or_not_lat = jump_or_not_lat, \
                    jerk_constraint_lon = jerk_constraint_lon, \
                    jerk_constraint_lat = jerk_constraint_lat, \
                    old_old_state = ego_state_previous, \
                    old_delta_sec = old_delta_sec )
                
                """
                #print(self.vehs_dict[vid], vid in F_dicts, vid in L_dicts)
                #
                #new_state = self.vehs_dict[vid] + F_dicts[vid] + np.matmul(L_dicts[vid], brownian/deltat_sec)
                #self.TrimUpdate(jerk_constraint = False, old_old_state = False, old_delta_sec = False)
                if jerk_constraint:
                    #TrimUpdate_two_dim_jump(self, old_state, deltat_sec, F, L, brownian, idm_paras = idm_paras, jumpsize_lon = .0, jumpsize_lat = .0, jump_or_not_lon = True, jump_or_not_lat = True, jerk_constraint = False, old_old_state = False, old_delta_sec = False)
                    new_state,feasible_increment = self.TrimUpdate_two_dim_jump(old_state = self.vehs_dict[vid], deltat_sec = deltat_sec, F = F_dicts[vid], L = L_dicts[vid], brownian = brownian, idm_paras = idm_paras, jumpsize_lon = jumpsize_lon, jumpsize_lat = jumpsize_lat,jump_or_not_lon = jump_or_not_lon, jump_or_not_lat = jump_or_not_lat, jerk_constraint = jerk_constraint, old_old_state = ego_state_previous, old_delta_sec = old_delta_sec )
                else:
                    new_state,feasible_increment = self.TrimUpdate_two_dim_jump(old_state = self.vehs_dict[vid], deltat_sec = deltat_sec, F = F_dicts[vid], L = L_dicts[vid], brownian = brownian, idm_paras = idm_paras, jumpsize = jumpsize_lon, jump_or_not = jump_or_not)
                
                """
                #
                #print(F_dicts[vid].shape, np.matmul(L_dicts[vid], brownian).shape)
                #new_state = self.vehs_dict[vid] + F_dicts[vid]*deltat_sec + np.matmul(L_dicts[vid], brownian)
                #---------------------UPDATE systen variables. 
                #SNAPSHOT
                self.snapshots[T]['vehs_at_lanes'] = copy.deepcopy(self.vehs_at_lanes)
                self.snapshots[T][vid] = {'S':copy.deepcopy(self.vehs_dict[vid]), \
                    'feasibleincrement':copy.deepcopy(feasible_increment), \
                    'L':copy.deepcopy(L_dicts[vid]), \
                    'F':copy.deepcopy(F_dicts[vid]), \
                    'potentials':copy.deepcopy(potentials_dict[vid]), }
                #
                #---------------------------------Update the vehicle order. 
                if new_state[0]>self.length:
                    new_state[0] = new_state[0] - self.length
                    #print(new_state.shape)
                    self.vehs_dict[vid] = new_state
                    #
                    tmp = copy.deepcopy([self.vehs_at_lanes[-1]] + self.vehs_at_lanes[:-1])
                    self.vehs_at_lanes = tmp
                else:
                    #print(new_state.shape)
                    self.vehs_dict[vid] = new_state
            #
            #############################RECORD the state
            STATES[T] = copy.deepcopy(self.vehs_dict)
            
            #
            builtins.tmp = STATES
        
        return STATES,irregular_vehs_ids



    def sim(self, deltat_sec =.5, T_horizon_sec = 3600.0, desired_density = 100, inserted_initial_state = np.array([.0, .0, .0, 0, .000001, .00001]), safety_gap2downstream_when_insert = 10, safety_gap2upstream_when_insert = 10,  idm_paras = idm_paras, stochastic_proecess_name = 'converted', two_dim_paras = two_dim_paras, leader_state_infinity = np.array([1e10, .0, .0, 0, .0, .0]), intert_tolerance = 1e-1, deltax_2d_or_not = False ):
        """
        
        """
        #STATES[t] = {vid:veh_state}
        STATES = {}
        self.snapshots = {}
        #
        T = 0
        while T<T_horizon_sec:
            self.snapshots[T] = {}
            ####################################Insert vehices 
            #obtain the densityed. lanesdensities[vid] = float, the density at the road. 
            density = self.get_lanes_densities()
            #
            if density>=desired_density:continue
            #
            if density>0:
                #
                vid_downstream =  self.vehs_at_lanes[0]
                vid_upstream = self.vehs_at_lanes[-1]
                if not (self.vehs_dict[vid_downstream][0]<=safety_gap2downstream_when_insert or self.length-self.vehs_dict[vid_upstream][0]<safety_gap2upstream_when_insert):
                    #INSERT VEHICLE.
                    vid_new = len(self.vehs_dict)#uuid.uuid1()
                    inserted_initial_state0 =  inserted_initial_state
                    inserted_initial_state0[0] = .0#np.random.uniform(0, safety_gap2downstream_when_insert)
                    inserted_initial_state0[2] = self.lw/2.0
                    #
                    self.vehs_dict[vid_new] = copy.deepcopy(inserted_initial_state0)
                    self.vehs_at_lanes.insert(0, vid_new)
                #
            else:
                #INSERT VEHICLE.
                vid_new = len(self.vehs_dict)#uuid.uuid1()
                inserted_initial_state0 =  inserted_initial_state
                inserted_initial_state0[0] = 0.0#np.random.uniform(0, safety_gap2downstream_when_insert)
                inserted_initial_state0[2] = self.lw/2.0
                #
                self.vehs_dict[vid_new] = copy.deepcopy(inserted_initial_state0)
                self.vehs_at_lanes.insert(0, vid_new)
            #
            #=======================================================
            #calcualte F and L. keys are the vehicle ids. 
            #   F_dicts[vid] is a 1d array. 
            #   L_dicts[vid] is a N*2 array. 2 means lon and lat noise. 
            F_dicts = {}
            L_dicts = {}
            leader_state_dict = {}
            potentials_dict = {}
            for i,vid in enumerate(self.vehs_at_lanes):
                # ----------------------------
                ego_state = self.vehs_dict[vid]
                #
                #------------------------------------------
                #find the leader 
                if len(self.vehs_at_lanes)==1:
                    leader_state = copy.deepcopy(ego_state)
                    leader_state[0] = ego_state[0]+1e10
                else:
                    if i==len(self.vehs_at_lanes)-1:
                        #
                        leader_id = self.vehs_at_lanes[0]
                        leader_state = copy.deepcopy(self.vehs_dict[leader_id])
                        leader_state[0] = leader_state[0]+ self.length
                        
                    else:
                        leader_id = self.vehs_at_lanes[i+1]
                        leader_state = self.vehs_dict[leader_id]
                    #------------------search leader at ego lane. 
                leader_state_dict[vid] = copy.deepcopy(leader_state)
                #
                #-----------------
                #--------------calculate the F and L
                #   F_dicts[vid] is a 1d array. 
                F_dicts[vid],potentials_dict[vid] = TwoDimMicroModel.F_SingleLane(ego_state  = ego_state, \
                    ego_lane_lw = self.lw, \
                    ego_lane_middle_line_coor = self.lw/2.0, \
                    ego_lane_marks_coor = (0, self.lw), \
                    road_bounds = (0, self.lw), \
                    lw_boundary_lanes = (self.lw, self.lw), \
                    leader_state = leader_state, \
                    two_dim_paras = two_dim_paras, \
                    stochastic_proecess_name = stochastic_proecess_name, deltax_2d_or_not = deltax_2d_or_not)
                L_dicts[vid] = TwoDimMicroModel.L(state  = ego_state, two_dim_paras = two_dim_paras, stochastic_proecess_name = stochastic_proecess_name)

                
            ################################UPDATE
            for vid in self.vehs_dict.keys():
                brownian = np.random.normal(loc = np.zeros((2,)), scale = deltat_sec)
                
                #
                #print(self.vehs_dict[vid], vid in F_dicts, vid in L_dicts)
                #
                #new_state = F_dicts[vid] + np.matmul(L_dicts[vid], brownian/deltat_sec)
                new_state,feasible_increment = self.TrimUpdate(old_state = self.vehs_dict[vid], deltat_sec = deltat_sec, F = F_dicts[vid], L = L_dicts[vid], brownian = brownian, idm_paras = idm_paras)
                #
                #print(F_dicts[vid].shape, np.matmul(L_dicts[vid], brownian).shape)
                #new_state = self.vehs_dict[vid] + F_dicts[vid]*deltat_sec + np.matmul(L_dicts[vid], brownian)
                #---------------------UPDATE systen variables. 
                #SNAPSHOT
                
                """
                
                self.snapshots[T][vid] = {'S':copy.deepcopy(self.vehs_dict[vid]), \
                    'feasibleincrement':copy.deepcopy(feasible_increment), \
                    'L':copy.deepcopy(L_dicts[vid]), \
                    'F':copy.deepcopy(F_dicts[vid]), \
                    'potentials':copy.deepcopy(potentials_dict[vid]), }
                """

                #
                
                if new_state[0]>self.length:
                    new_state[0] = new_state[0] - self.length
                    #print(new_state.shape)
                    self.vehs_dict[vid] = new_state
                    #
                    tmp = copy.deepcopy([self.vehs_at_lanes[-1]] + self.vehs_at_lanes[:-1])
                    self.vehs_at_lanes = tmp
                else:
                    #print(new_state.shape)
                    self.vehs_dict[vid] = new_state
            #
            #############################RECORD the state
            STATES[T] = copy.deepcopy(self.vehs_dict)
            
            #
            T = T + deltat_sec
            builtins.tmp = STATES
        
        return STATES








class TwoDimMicroModel():
    """
    two dim microscopic traffic flow model. 
    
    The state of the vehicle is represented as:
    
        - x,y,vx,vy,zlon,zlat.  x is the longitudinal, y is the lateral, z is the noise. 
    
    """

    @classmethod
    def SpeedDependentJumpAdjust_lat(self, v_ms):
        """
        Difference:
        
            - self.SpeedDependentJumpAdjust
            - self.SpeedDependentJumpAdjust_lat
            
        The latter one is for the lateral dimension.
        
        
        Adjust the jump size, which depends on the speed v. 
        Callback: 
        
            reload(irre)
            vs = np.linspace(1, 120, 200)
            res = [irre.TwoDimMicroModel.SpeedDependentJumpAdjust(v) for v in vs]
            plt.plot(vs, res)
                    
        Run the following to see the speed-dependent curve:
        
            import stats
            vs = np.linspace(0, 80, 100)
            ys = [150*scipy.stats.lognorm(s = 1, scale=math.exp(3)).pdf(v) for v in vs]
            plt.plot(vs,ys)
            
            res = TwoDimMicroModel.SpeedDependentJumpAdjust(v_ms)
        
        -----------------------------
        @input: v   
        
            a float. unit is m/s
        
        @OUTPUT: scale
        
            
            a float represent the scale.
        
            
        """
        #10 max
        return 150*scipy.stats.lognorm(s = 1, scale=math.exp(3.2)).pdf(v_ms)
        
        
        pass


    @classmethod
    def SpeedDependentJumpAdjust(self, v_ms):
        """
        Adjust the jump size, which depends on the speed v. 
        Callback: 
        
            reload(irre)
            vs = np.linspace(1, 120, 200)
            res = [irre.TwoDimMicroModel.SpeedDependentJumpAdjust(v) for v in vs]
            plt.plot(vs, res)
                    
        Run the following to see the speed-dependent curve:
        
            import stats
            vs = np.linspace(0, 80, 100)
            ys = [150*scipy.stats.lognorm(s = 1, scale=math.exp(3)).pdf(v) for v in vs]
            plt.plot(vs,ys)
            
            res = TwoDimMicroModel.SpeedDependentJumpAdjust(v_ms)
        
        -----------------------------
        @input: v   
        
            a float. unit is m/s
        
        @OUTPUT: scale
        
            
            a float represent the scale.
        
            
        """
        #10 max
        return 150*scipy.stats.lognorm(s = 1, scale=math.exp(3.2)).pdf(v_ms)
        
        
        pass

    @classmethod
    def Get_QVK_from_snapshots_singlelane_jump(self, snapshots, length, Deltat_FD_sec = 30, rolling_sec = 30,):
        """
        DIfference between:
        
            - self.Get_QVK_from_snapshots()
            - self.Get_QVK_from_snapshots_singlelane()
            
        The latter is for single lane simulation. The snapshot is defined via:
        
        
                self.snapshots[T][vid] = {'S':copy.deepcopy(self.vehs_dict[vid]), \
                    'feasibleincrement':copy.deepcopy(feasible_increment), \
                    'L':copy.deepcopy(L_dicts[vid]), \
                    'F':copy.deepcopy(F_dicts[vid]), \
                    'potentials':copy.deepcopy(potentials_dict[vid]), \
                    'vehs_at_lanes':copy.deepcopy(self.vehs_at_lanes),}
        
        
        calculate the Q and K
        
        ------------------------------------------------------------------
        
        @input: snapshots
        
                self.snapshots[T][vid] = {'S':copy.deepcopy(self.vehs_dict[vid]), \
                    'feasibleincrement':copy.deepcopy(feasible_increment), \
                    'L':copy.deepcopy(L_dicts[vid]), \
                    'F':copy.deepcopy(F_dicts[vid]), \
                    'potentials':copy.deepcopy(potentials_dict[vid]), \
                    'vehs_at_lanes':copy.deepcopy(self.vehs_at_lanes),}
        
        @input: Deltat_FD_sec
        
            the time step that calculate the FD parameters. 
        
        @OUTPUT: Q and K
        
            both are dicts. 
            
            Q = [q1, q2, q3, q4...]
            K = [k1, k2, k3, k4...]
            V = [v1,v2,....]
            Q unit is veh/h
            K unit is veh/km
        
        
        
        
        -----------------------------
        Q = dA/|A|, d is the distance travelled by all vehicles.
        K = t(A)/|A|, t is the total time travelled. 
        """
        ts = sorted(snapshots.keys())
        #Q[laneid] = [q1, q2, q3, q4...]
        #K[laneid] = [k1, k2, k3, k4...]
        t = sorted(snapshots.keys())[0]
        Q = []
        K = []
        V = []
        
        #unit is km.h
        #area = (length/1000.0)*(Deltat_FD_sec/3600.0)
        #
        Ts = np.array(sorted(snapshots.keys()))
        #
        start = 0.0
        end = start + Deltat_FD_sec
        for t in Ts:
            #for lane_id in snapshots[t]['vehs_at_lanes'].keys():
            k = len(snapshots[t])/(length/1000.0)
            K.append(k)
            v =  3.6*np.mean([snapshots[t][vid]['S'][1] for vid in snapshots[t].keys()])
            V.append(v)
            Q.append(k*v)
            
        return Q,V,K
        
        
        
        
        
        
        
        while end<=max(Ts):
            #unit is sec and meter.
            totaltimetravelled_sec = 0
            totaldistancetravelled_meter = 0
            #
            interval_ts = sorted(Ts[(Ts>start) & (Ts<=end)])
            for t0,t1 in zip(interval_ts[:-1], interval_ts[1:]):
                #
                for lane_id in snapshots[t]['vehs_at_lanes'].keys():
                    #
                    for vid in snapshots[t]['vehs_at_lanes'][lane_id]:
                        #time travelled
                        totaltimetravelled_sec = totaltimetravelled_sec + t1 - t0
                        #distancetravelled
                        if snapshots[t1][vid]['S'][0]<=snapshots[t0][vid]['S'][0]:
                            distance_travelled_vid = snapshots[t1][vid]['S'][0] + length - snapshots[t0][vid]['S'][0]
                        else:
                            distance_travelled_vid = snapshots[t1][vid]['S'][0]  - snapshots[t0][vid]['S'][0]
                        #
                        totaldistancetravelled_meter = totaldistancetravelled_meter  + distance_travelled_vid
                #
                Q_interval = totaldistancetravelled_meter/1000.0/area
                K_interval = totaldistancetravelled_meter/3600.0/area
                V_interval = Q_interval/K_interval
                #
                Q[lane_id].append(Q_interval)
                K[lane_id].append(K_interval)
                V[lane_id].append(V_interval)
            start = start + rolling_sec
            end = start + Deltat_FD_sec
        
        return Q,V,K
        


    @classmethod
    def Get_QVK_from_snapshots_singlelane(self, snapshots, length, Deltat_FD_sec = 30, rolling_sec = 30,):
        """
        DIfference between:
        
            - self.Get_QVK_from_snapshots()
            - self.Get_QVK_from_snapshots_singlelane()
            
        The latter is for single lane simulation. The snapshot is defined via:
        
        
                self.snapshots[T][vid] = {'S':copy.deepcopy(self.vehs_dict[vid]), \
                    'feasibleincrement':copy.deepcopy(feasible_increment), \
                    'L':copy.deepcopy(L_dicts[vid]), \
                    'F':copy.deepcopy(F_dicts[vid]), \
                    'potentials':copy.deepcopy(potentials_dict[vid]), \
                    'vehs_at_lanes':copy.deepcopy(self.vehs_at_lanes),}
        
        
        calculate the Q and K
        
        ------------------------------------------------------------------
        
        @input: snapshots
        
                self.snapshots[T][vid] = {'S':copy.deepcopy(self.vehs_dict[vid]), \
                    'feasibleincrement':copy.deepcopy(feasible_increment), \
                    'L':copy.deepcopy(L_dicts[vid]), \
                    'F':copy.deepcopy(F_dicts[vid]), \
                    'potentials':copy.deepcopy(potentials_dict[vid]), \
                    'vehs_at_lanes':copy.deepcopy(self.vehs_at_lanes),}
        
        @input: Deltat_FD_sec
        
            the time step that calculate the FD parameters. 
        
        @OUTPUT: Q and K
        
            both are dicts. 
            
            Q = [q1, q2, q3, q4...]
            K = [k1, k2, k3, k4...]
            V = [v1,v2,....]
            Q unit is veh/h
            K unit is veh/km
        
        
        
        
        -----------------------------
        Q = dA/|A|, d is the distance travelled by all vehicles.
        K = t(A)/|A|, t is the total time travelled. 
        """
        #Q[laneid] = [q1, q2, q3, q4...]
        #K[laneid] = [k1, k2, k3, k4...]
        t = sorted(snapshots.keys())[0]
        Q = []
        K = []
        V = []
        
        #unit is km.h
        #area = (length/1000.0)*(Deltat_FD_sec/3600.0)
        #
        Ts = np.array(sorted(snapshots.keys()))
        #
        start = 0.0
        end = start + Deltat_FD_sec
        for t in Ts:
            #for lane_id in snapshots[t]['vehs_at_lanes'].keys():
            k = len(snapshots[t]['vehs_at_lanes'])/(length/1000.0)
            K.append(k)
            v =  3.6*np.mean([snapshots[t][vid]['S'][1] for vid in snapshots[t]['vehs_at_lanes']])
            V.append(v)
            Q.append(k*v)
            
        return Q,V,K
        
        
        
        
        
        
        
        while end<=max(Ts):
            #unit is sec and meter.
            totaltimetravelled_sec = 0
            totaldistancetravelled_meter = 0
            #
            interval_ts = sorted(Ts[(Ts>start) & (Ts<=end)])
            for t0,t1 in zip(interval_ts[:-1], interval_ts[1:]):
                #
                for lane_id in snapshots[t]['vehs_at_lanes'].keys():
                    #
                    for vid in snapshots[t]['vehs_at_lanes'][lane_id]:
                        #time travelled
                        totaltimetravelled_sec = totaltimetravelled_sec + t1 - t0
                        #distancetravelled
                        if snapshots[t1][vid]['S'][0]<=snapshots[t0][vid]['S'][0]:
                            distance_travelled_vid = snapshots[t1][vid]['S'][0] + length - snapshots[t0][vid]['S'][0]
                        else:
                            distance_travelled_vid = snapshots[t1][vid]['S'][0]  - snapshots[t0][vid]['S'][0]
                        #
                        totaldistancetravelled_meter = totaldistancetravelled_meter  + distance_travelled_vid
                #
                Q_interval = totaldistancetravelled_meter/1000.0/area
                K_interval = totaldistancetravelled_meter/3600.0/area
                V_interval = Q_interval/K_interval
                #
                Q[lane_id].append(Q_interval)
                K[lane_id].append(K_interval)
                V[lane_id].append(V_interval)
            start = start + rolling_sec
            end = start + Deltat_FD_sec
        
        return Q,V,K
        



    @classmethod
    def Get_QVK_from_snapshots(self, snapshots, length, Deltat_FD_sec = 30, rolling_sec = 30,):
        """
        calculate the Q and K
        
            = 
        
        @input: snapshots
        
            snapshots[moment].kesys() are:
            
                self.snapshots[T]['vehs_at_lanes'] = copy.deepcopy(self.vehs_at_lanes)
                self.snapshots[T][vid] = {'S':copy.deepcopy(self.vehs_dict[vid]), \
                    #'feasibleincrement':copy.deepcopy(feasible_increment), \
                    'L':copy.deepcopy(L_dicts[vid]), \
                    'F':copy.deepcopy(F_dicts[vid]), \
                    'vehs_info_within_area_left':copy.deepcopy(vehs_info_within_area_left_dict[vid]), \
                    'vehs_info_within_area_right':copy.deepcopy(vehs_info_within_area_right_dict[vid]), \
                    'potentials':copy.deepcopy(potentials_dict[vid]), }
        
        @input: Deltat_FD_sec
        
            the time step that calculate the FD parameters. 
        
        @OUTPUT: Q and K
        
            both are dicts. 
            
            Q[laneid] = [q1, q2, q3, q4...]
            K[laneid] = [k1, k2, k3, k4...]
            V[laneid] = [v1,v2,....]
            Q unit is veh/h
            K unit is veh/km
        
        
        
        
        -----------------------------
        Q = dA/|A|, d is the distance travelled by all vehicles.
        K = t(A)/|A|, t is the total time travelled. 
        """
        #Q[laneid] = [q1, q2, q3, q4...]
        #K[laneid] = [k1, k2, k3, k4...]
        t = sorted(snapshots.keys())[0]
        Q = {lane_id:[] for lane_id in snapshots[t]['vehs_at_lanes'].keys()}
        K = {lane_id:[] for lane_id in snapshots[t]['vehs_at_lanes'].keys()}
        V = {lane_id:[] for lane_id in snapshots[t]['vehs_at_lanes'].keys()} 
        
        #unit is km.h
        area = (length/1000.0)*(Deltat_FD_sec/3600.0)
        #
        Ts = np.array(sorted(snapshots.keys()))
        #
        start = 0.0
        end = start + Deltat_FD_sec
        for t in Ts:
            for lane_id in snapshots[t]['vehs_at_lanes'].keys():
                k = len(snapshots[t]['vehs_at_lanes'][lane_id])/(length/1000.0)
                K[lane_id].append(k)
                #[1] is because the state variable is [x, vx, y ,vy, zlon, zlat]
                v =  3.6*np.mean([snapshots[t][vid]['S'][1] for vid in snapshots[t]['vehs_at_lanes'][lane_id]])
                V[lane_id].append(v)
                Q[lane_id].append(k*v)
                
        return Q,V,K
        
        
        
        while end<=max(Ts):
            #unit is sec and meter.
            totaltimetravelled_sec = 0
            totaldistancetravelled_meter = 0
            #
            interval_ts = sorted(Ts[(Ts>start) & (Ts<=end)])
            for t0,t1 in zip(interval_ts[:-1], interval_ts[1:]):
                #
                for lane_id in snapshots[t]['vehs_at_lanes'].keys():
                    #
                    for vid in snapshots[t]['vehs_at_lanes'][lane_id]:
                        #time travelled
                        totaltimetravelled_sec = totaltimetravelled_sec + t1 - t0
                        #distancetravelled
                        if snapshots[t1][vid]['S'][0]<=snapshots[t0][vid]['S'][0]:
                            distance_travelled_vid = snapshots[t1][vid]['S'][0] + length - snapshots[t0][vid]['S'][0]
                        else:
                            distance_travelled_vid = snapshots[t1][vid]['S'][0]  - snapshots[t0][vid]['S'][0]
                        #
                        totaldistancetravelled_meter = totaldistancetravelled_meter  + distance_travelled_vid
                #
                Q_interval = totaldistancetravelled_meter/1000.0/area
                K_interval = totaldistancetravelled_meter/3600.0/area
                V_interval = Q_interval/K_interval
                #
                Q[lane_id].append(Q_interval)
                K[lane_id].append(K_interval)
                V[lane_id].append(V_interval)
            start = start + rolling_sec
            end = start + Deltat_FD_sec
        
        return Q,V,K
        

    @classmethod
    def Get_QVK_from_snapshots_edie(self, snapshots, length, Deltat_FD_sec = 30, rolling_sec = 30,):
        """
        calculate the Q and K
        
            = 
        
        @input: snapshots
        
            snapshots[moment].kesys() are:
            
                self.snapshots[T]['vehs_at_lanes'] = copy.deepcopy(self.vehs_at_lanes)
                self.snapshots[T][vid] = {'S':copy.deepcopy(self.vehs_dict[vid]), \
                    #'feasibleincrement':copy.deepcopy(feasible_increment), \
                    'L':copy.deepcopy(L_dicts[vid]), \
                    'F':copy.deepcopy(F_dicts[vid]), \
                    'vehs_info_within_area_left':copy.deepcopy(vehs_info_within_area_left_dict[vid]), \
                    'vehs_info_within_area_right':copy.deepcopy(vehs_info_within_area_right_dict[vid]), \
                    'potentials':copy.deepcopy(potentials_dict[vid]), }
        
        @input: Deltat_FD_sec
        
            the time step that calculate the FD parameters. 
        
        @OUTPUT: Q and K
        
            both are dicts. 
            
            Q[laneid] = [q1, q2, q3, q4...]
            K[laneid] = [k1, k2, k3, k4...]
            V[laneid] = [v1,v2,....]
            Q unit is veh/h
            K unit is veh/km
        
        
        
        
        -----------------------------
        Q = dA/|A|, d is the distance travelled by all vehicles.
        K = t(A)/|A|, t is the total time travelled. 
        """
        #Q[laneid] = [q1, q2, q3, q4...]
        #K[laneid] = [k1, k2, k3, k4...]
        t = sorted(snapshots.keys())[0]
        Q = {lane_id:[] for lane_id in snapshots[t]['vehs_at_lanes'].keys()}
        K = {lane_id:[] for lane_id in snapshots[t]['vehs_at_lanes'].keys()}
        V = {lane_id:[] for lane_id in snapshots[t]['vehs_at_lanes'].keys()} 
        
        #unit is km.h
        area = (length/1000.0)*(Deltat_FD_sec/3600.0)
        #
        Ts = np.array(sorted(snapshots.keys()))
        #
        start = 0.0
        end = start + Deltat_FD_sec
        while end<=max(Ts):
            #unit is sec and meter.
            totaltimetravelled_sec = 0
            totaldistancetravelled_meter = 0
            #
            interval_ts = sorted(Ts[(Ts>start) & (Ts<=end)])
            for t0,t1 in zip(interval_ts[:-1], interval_ts[1:]):
                #
                for lane_id in snapshots[t]['vehs_at_lanes'].keys():
                    #
                    for vid in snapshots[t]['vehs_at_lanes'][lane_id]:
                        #time travelled
                        totaltimetravelled_sec = totaltimetravelled_sec + t1 - t0
                        #distancetravelled
                        if snapshots[t1][vid]['S'][0]<=snapshots[t0][vid]['S'][0]:
                            distance_travelled_vid = snapshots[t1][vid]['S'][0] + length - snapshots[t0][vid]['S'][0]
                        else:
                            distance_travelled_vid = snapshots[t1][vid]['S'][0]  - snapshots[t0][vid]['S'][0]
                        #
                        totaldistancetravelled_meter = totaldistancetravelled_meter  + distance_travelled_vid
                #
                Q_interval = totaldistancetravelled_meter/1000.0/area
                K_interval = totaldistancetravelled_meter/3600.0/area
                V_interval = Q_interval/K_interval
                #
                Q[lane_id].append(Q_interval)
                K[lane_id].append(K_interval)
                V[lane_id].append(V_interval)
            start = start + rolling_sec
            end = start + Deltat_FD_sec
        
        return Q,V,K
        


    @classmethod
    def interpolate(self, es, hs, e, outofbound_returned = 0):
        """
        h = self.interpolate(es,hs,e)
        
        """
        if e<min(es) or e>max(es):
            return outofbound_returned
        
        #==============================
        
        #find the idx that es[idx]<=e<=es[idx+1]
        all_idxs = np.array(range(len(es)))
        idxs_equal = all_idxs[es==e]
        if len(idxs_equal)>0:
            return hs[idxs_equal[0]]
        #
        idx = all_idxs[es<e][-1]
        
        #interploate
        #print(idx+1, len(hs), e, max(es), es[-2], es[-1])
        return (hs[idx] + hs[idx+1])/2.0
        
    
    @classmethod
    def ConvertLateralDistribution2ESAL(self, edges, hists, vehiclewidth = 1.9):
        """
        edges_new,hists_new = self.ConvertLateralDistribution2ESAL(edges, hists, vehiclewidth)
        -------------------------------
        @input: vehiclewidth
        
            unit is m. 
            
        @input: edges, hists
        
            both are arrya. 
        @output:
        
            edgesnew,histnew
            
        """
        #=========================================================
        edges_left = np.array(edges) - vehiclewidth/2.0
        edges_right = np.array(edges) + vehiclewidth/2.0 
        #
        edges_new = sorted(list(edges_left) + list(edges_right) )
        hists_new = [(self.interpolate(edges_left,hists,e) + self.interpolate(edges_right,hists,e))/2.0 for e in edges_new]
        #
        return edges_new,hists_new
        
        
        pass
    
    
    @classmethod
    def TrimAcce(self, acce, STATE, idm_paras = idm_paras):
        """
        Change the acceleraiton to make sure that the resulting vehicle constraints would not be violated. 
        
        The following conditions are considered when trimming:
        
            - if the speed reaches the maximum or the minimum, the acceleration is zero
            - if the 
        
        -----------------------------
        @OUTPUT acce_trimmed
        
            the trimmed acceleration. 
            
            
            
        """
        
        #the system  state. 
        x0,vx0,y,vy,Z_long,Z_lat = STATE[0],STATE[1],STATE[2],STATE[3],STATE[4],STATE[5]
        #x0,y,phi,v0,delta,Z_long,Z_lat = STATE_init[0],STATE_init[1],STATE_init[2],STATE_init[3],STATE_init[4],STATE_init[5],STATE_init[6]
        
        #the system  state. 
        #x,y,phi,v,delta,Z_long,Z_lat = STATE[0],STATE[1],STATE[2],STATE[3],STATE[4],STATE[5],STATE[6]
        
        #
        #if v<0 or v>=1.3*idm_paras['idm_vf']:
        if vx0<.0:
            return 1.0
        if vx0>idm_paras['idm_vf']:
            return -1
            
        #-------------------------
        return min(idm_paras['idm_a'], max(-idm_paras['idm_b'], acce))

    
    @classmethod
    def TransformPeriodic_ys_2MultiSegment_return_idxs_segments(self, ys, length,):
        """
        idxs_segments = TransformPeriodic_ys_2MultiSegment_return_idxs_segments(ys, length)
        ---------------------------
        The trajectory is period means that the vehicle will return to 0 when they are to exit the 
        
        @input: ys
        
            a 1d array that describe the longitudinal coordinate. 
            
        @input: length
        
            the length of the road
        
        -----------------------------------------------
        @OUTPUT: idxs_segments
            
            list of list
        
        ---------------------------------------------
        @Steps:
        
            - Find the idx that is decreasing
            - Add the length.
        
        """
        idxs_segments = []
        #Step1: find the index that decreeasing, i.e. ys[idx]>ys[idx-1]
        idxs_decreasing = np.where(np.diff(ys)<0)[0]
        #print(idxs_decreasing)
        #
        if len(idxs_decreasing)==0:
            #print(ys)
            return [list(range(len(ys)))]
        ##########################################ys_es
        #Note the '+1'
        #ys_es = [ys[:idxs_decreasing[0]+1]]
        
        idxs_segments = [range(idxs_decreasing[0]+1)]
        
        for idx_in_decreasing,idx_y in enumerate(idxs_decreasing):
            if idx_in_decreasing==0:continue
            #
            idx_start_in_ys = idxs_decreasing[idx_in_decreasing-1] + 1
            idx_end_in_ys = idx_y + 1
            idxs_segments.append(range(idx_start_in_ys, idx_end_in_ys))
            #
            #ys_after_add_length = ys[idx_start_in_ys:idx_end_in_ys]# + length*(idx_in_decreasing + 1))
            #ys_after_add_length = list(np.array(ys[idx_start_in_ys:idx_end_in_ys]))# + length*(idx_in_decreasing + 1))
            #
            #ys_es.append(ys_after_add_length)
            #
        #THE LAST segment
        idx_start_in_ys = idxs_decreasing[-1] + 1
        #ys_after_add_length = ys[idx_start_in_ys:]
        idxs_segments.append(range(idx_start_in_ys, len(ys)))
        return idxs_segments
        
        """

        
        """
        
        
        
    @classmethod
    def random_accelerations(self, amplification = 3.0, ts = np.linspace(0, 300, 300), period_sec = 30):
        """
        
        
        acces  = VM.TwoDimStochasticIDM.random_accelerations(ts = ts)
        """
        return np.array([amplification*np.sin(t/period_sec) for t in ts[1:]])


    @classmethod
    def GenerateLeaderTrajectories(self, ts = np.linspace(0, 300, 300), STATE_init = np.array([.0, .0, .0, .0, .0, .0]), idm_paras  = idm_paras, ):
        """
        Generate the trajectories of the leader. 
        
        Each instance, the state of the vehicle is represented by STATES:
        
        dS = F(S) dt + L(S) dW
        
        @input:     
            
            The state of the vehicle is represented as:
        
            - x,y,vx,vy,zlon,zlat.  x is the longitudinal, y is the lateral, z is the noise. 
            
        Note that in the leading trajectory we only consider the longitudinal (x) dimeensnon. 
    
        
        NOTE the coordinate system: 
        
            that the x is the logitudinal axis and y is the lateral. 
        
            The poisitive axis it downstream. 
        
            The left direction is y-positive. 
            
            If the vehicle turn left the steeer angle is positive. 
        
        
        -----------------------------------------------------------------
        
        @input: STATE_init
        
            the initial state. 
        
        @input: ts
        
            the moments. 
        
        @OUTPUT: trajectories
        
            an np.array. 
            
            Shape is (7, moments_N), where 7 is the state number, and moments_N is the number of moments. 
        
            
        
        """
        
        #the system  state. 
        x0,vx0,y,vy,Z_long,Z_lat = STATE_init[0],STATE_init[1],STATE_init[2],STATE_init[3],STATE_init[4],STATE_init[5]
        #x0,y,phi,v0,delta,Z_long,Z_lat = STATE_init[0],STATE_init[1],STATE_init[2],STATE_init[3],STATE_init[4],STATE_init[5],STATE_init[6]
        
        #the length of accelerations is len(ts-1).
        accelerations = self.random_accelerations(ts = ts)
        
        #IDM paraser. 
        idm_vf = idm_paras['idm_vf']
        idm_T = idm_paras['idm_T']
        idm_delta = idm_paras['idm_delta']
        idm_s0 = idm_paras['idm_s0']
        idm_a = idm_paras['idm_a']
        idm_b = idm_paras['idm_b']
        
        #find the xs
        xs = [x0]
        vs = [vx0]
        for deltat,acc0 in zip(np.diff(ts), accelerations):
            STATES_tmp = np.array([xs[-1], vs[-1],  y, vy, Z_long,Z_lat])
            acc= self.TrimAcce(acc0, STATE = STATES_tmp, idm_paras = idm_paras)
            #
            #print(acc0, acc)
            new_v = min(max(0, vs[-1] + deltat*acc), idm_vf)
            #
            new_x = xs[-1] + deltat*vs[-1]
            
            #
            vs.append(new_v)
            xs.append(new_x)
            
        #
        
        return np.array([xs, vs, [y]*len(ts),[0]*len(ts),[Z_long]*len(ts),[Z_lat]*len(ts)])
        
    
    
    @classmethod
    def PotentialRoadBoundary(self,  ego_state, road_bounds, lw_boundary_lanes = [3.5, 3.5], two_dim_paras = two_dim_paras, right_force_exist = True, left_force_exist = True, ):
        """
        
        @input: ego_state
        
            ego_state
        
            a  state of the ego vehicle. 
            
            
            The state of the vehicle is represented as:
        
            - np.array([x, vx, y, vy, zlon, zlat])  x is the longitudinal, y is the lateral, z is the noise. 
        
        @input: lw_boundary_lanes
        
            leftmost lane width and rightmost lane width. 
            
            THey should corresponds to the arg road_bounds.
        
        @input: road_bounds
        
            road_bounds = (float0 ,float1)
        
        @OUTPUT: potential_road_boundary
        
            a float. 
        
        """
        #
        x,vx,y, vy,Z_long,Z_lat = ego_state[0],ego_state[1],ego_state[2],ego_state[3],ego_state[4],ego_state[5]
        #
        #   the parameter. 
        alpha_roadbounds =  two_dim_paras['alpha_roadbounds']
        #
        #potential0 must be negative. 
        deltay = y-road_bounds[0]
        deltay0 = max(1e-10, min(abs(deltay), lw_boundary_lanes[0]/2.0))
        #print(y, road_bounds[0])
        #deltay>0 means the right force
        if abs(deltay)<lw_boundary_lanes[0]/2.0:
            if deltay>0:
                if right_force_exist:
                    potential0 = 1.0/(np.power(deltay0*2.0/lw_boundary_lanes[0], alpha_roadbounds)) - 1
                else:
                    potential0 = .0
            else:
                if left_force_exist:
                    potential0 = -(1.0/(np.power(deltay0*2.0/lw_boundary_lanes[0], alpha_roadbounds)) - 1)
                else:
                    potential0 = .0
        else:
            potential0 = .0
        #
        deltay = y-road_bounds[1]
        deltay1 = max(1e-10, min(abs(deltay), lw_boundary_lanes[1]/2.0))
        #print(deltay1*2.0/lw_boundary_lanes[1], alpha_roadbounds)
        if abs(deltay)<lw_boundary_lanes[1]/2.0:
            if deltay>0:
                if right_force_exist:
                    potential1 = 1.0/(np.power(deltay1*2.0/lw_boundary_lanes[1], alpha_roadbounds)) - 1
                else:
                    potential1 = .0
            else:
                #print(deltay1)
                if left_force_exist:
                    potential1 = -(1.0/(np.power(deltay1*2.0/lw_boundary_lanes[1], alpha_roadbounds)) -1)
                else:
                    potential1 = .0
        else:
            potential1 = .0
        #
        #
        return potential0 + potential1
        
        ############################################################################################
        ############################################################################################
        ############################################################################################
        #
        #x,vx,y, vy,Z_long,Z_lat = ego_state[0],ego_state[1],ego_state[2],ego_state[3],ego_state[4],ego_state[5]
        #
        #   the parameter. 
        alpha_roadbounds =  two_dim_paras['alpha_roadbounds']
        #
        #========================================================the first bound
        #potential0 must be negative. 
        deltay = y-road_bounds[0]
        deltay0 = max(1e-10, abs(deltay))
        #print(y, road_bounds[0])
        if abs(deltay)<lw_boundary_lanes[0]/2.0:
            if deltay>0:
                #deltay>0 means the right force
                if right_force_exist:
                    #potential0 = 1.0/(np.power(deltay0*2.0/ego_lane_lw, alpha_roadbounds)) - 1
                    potential0 = 1.0/(np.power(deltay0*2.0/ego_lane_lw, alpha_roadbounds)) - 1
                else:
                    potential0 = 0.0
                    
            else:
                if left_force_exist:
                    #potential0 = -(1.0/(np.power(deltay0*2.0/ego_lane_lw, alpha_roadbounds)) - 1)
                    potential0 = -(1.0/(np.power(deltay0*2.0/ego_lane_lw, alpha_roadbounds)) - 1)
                else:
                    potential0 = 0
        else:
            potential0 = .0
        #========================================================the second bound
        deltay = y-road_bounds[1]
        deltay1 = max(1e-10, abs(deltay))
        if abs(deltay)<lw_boundary_lanes[1]/2.0:
            if deltay>0:
                #deltay>0 means the right force
                if right_force_exist:
                    #potential1 = 1.0/(np.power(deltay1*2.0/ego_lane_lw, alpha_roadbounds)) - 1
                    potential1 = 1.0/(np.power(deltay1*2.0/ego_lane_lw, alpha_roadbounds)) - 1
                else:
                    potential1 = 0.0
            else:
                if left_force_exist:
                    #print(deltay1)
                    #potential1 = -(1.0/(np.power(deltay1*2.0/ego_lane_lw, alpha_roadbounds)) -1)
                    potential1 = -(1.0/(np.power(deltay1*2.0/ego_lane_lw, alpha_roadbounds)) -1)
                else:
                    potential1 = 0.0
        else:
            potential1 = 0.0
        
        return potential0 + potential1
        
        
    
    @classmethod
    def PotentialNeighboringLongitudinalLateral_BKP(self, ego_state, ego_lane_lw, vehs_info_within_area_left, vehs_info_within_area_right, ellipse_x, ellipse_y_left, ellipse_y_right):
        """
        Callback
        
            longitudinalforce,lateralforce = self.PotentialNeighboringLongitudinalLateral(ego_state, ego_lane_lw, vehs_ids_within_area, ellipse_x, ellipse_y)
        
        ---------------------------------------------------------------
        @input: ego_state
        
            ego_state = np.array([x, vx, y, vy, zlon, zlat])
        
        @input: vehs_ids_within_area
            
            the keys are vids. 
            
            vehs_ids_within_area[vid] = np.array([x, vx, y, vy, zlon, zlat])
            
            vehs_ids_within_area = {}
        
        @OUTPUT: longitudinalforce,lateralforce
        
            two floats. 
            
            The sign indicate the direction of the force. 
            
            negative means that the force points to the negative direction of the y axis. 
        
        --------------------------------------------------------
        
        """
        longitudinalforce = 0
        lateralforce = 0
        #######################################LEFT
        for vid in vehs_info_within_area_left.keys():
            #
            #============================total force
            neighbor_state = vehs_info_within_area_left[vid]
            #
            deltax = ego_state[0] - neighbor_state[0]
            deltay = ego_state[2] - neighbor_state[2]
            #totalforce
            tmp0  = -(ego_state[0]**2)/2.0*abs(deltax)
            #   tmp1 must be positive, as abs(deltay) must be smaller than 
            tmp1 = 1.0/(abs(deltay) - ego_lane_lw/2.0)
            #   tmp2 MUST BE positive. 
            tmp2 = 1.0/(ellipse_y_left*np.sqrt(1.0 - (deltax**2)/(ellipse_x**2)) - ego_lane_lw/2.0)
            #
            #--------------------total force
            totalforce = abs(tmp0*(tmp1 - tmp2))
            #
            #====================================longitudinalforce, MUST BE NEGATIVE. 
            #print(deltax, deltay)
            longitudinalforce = longitudinalforce - totalforce*abs(deltax)/(np.sqrt(deltax**2 + deltay**2))
            #--------------------------------------
            lateralforce = lateralforce -totalforce*abs(deltay)/(np.sqrt(deltax**2 + deltay**2))
            """
            if ego_state[2] > neighbor_state[2]:
                lateralforce = lateralforce + totalforce*abs(deltay)/(np.sqrt(deltax**2 + deltay**2))
            else:
                lateralforce = lateralforce -totalforce*abs(deltay)/(np.sqrt(deltax**2 + deltay**2))
            """
        #######################################RIGHT
        for vid in vehs_info_within_area_right.keys():
            #
            #============================total force
            neighbor_state = vehs_info_within_area_right[vid]
            #
            deltax = ego_state[0] - neighbor_state[0]
            deltay = ego_state[2] - neighbor_state[2]
            #totalforce
            tmp0  = -(ego_state[0]**2)/2.0*abs(deltax)
            #   tmp1 must be positive, as abs(deltay) must be smaller than 
            tmp1 = 1.0/(abs(deltay) - ego_lane_lw/2.0)
            #   tmp2 MUST BE positive. 
            tmp2 = 1.0/(ellipse_y_right*np.sqrt(1.0 - (deltax**2)/(ellipse_x**2)) - ego_lane_lw/2.0)
            #
            #--------------------total force
            totalforce = abs(tmp0*(tmp1 - tmp2))
            #
            #====================================longitudinalforce, MUST BE NEGATIVE. 
            longitudinalforce = longitudinalforce - totalforce*abs(deltax)/(np.sqrt(deltax**2 + deltay**2))
            #
            lateralforce = lateralforce + totalforce*abs(deltay)/(np.sqrt(deltax**2 + deltay**2))
            #--------------------------------------
            """
            if ego_state[2] > neighbor_state[2]:
                lateralforce = lateralforce + totalforce*abs(deltay)/(np.sqrt(deltax**2 + deltay**2))
            else:
                lateralforce = lateralforce -totalforce*abs(deltay)/(np.sqrt(deltax**2 + deltay**2))
            """
        
        return longitudinalforce,lateralforce
    

    
    @classmethod
    def PotentialNeighboringLongitudinalLateral_lw_base(self, ego_state, ego_lane_lw, \
            vehs_info_within_area_left, \
            vehs_info_within_area_right, \
            ellipse_x, ellipse_y_left, ellipse_y_right, \
            force_scale_deletable = 1.0, \
            lw_base = 3.5):
        """
        Callback
        
            longitudinalforce,lateralforce = self.PotentialNeighboringLongitudinalLateral(ego_state, ego_lane_lw, vehs_ids_within_area, ellipse_x, ellipse_y)
        
        ---------------------------------------------------------------
        @input: ego_state
        
            ego_state = np.array([x, vx, y, vy, zlon, zlat])
        
        @input: vehs_ids_within_area
            
            the keys are vids. 
            
            vehs_ids_within_area[vid] = np.array([x, vx, y, vy, zlon, zlat])
            
            vehs_ids_within_area = {}
        
        @OUTPUT: longitudinalforce,lateralforce
        
            two floats. 
            
            The sign indicate the direction of the force. 
            
            negative means that the force points to the negative direction of the y axis. 
        
        --------------------------------------------------------
        
        """
        #
        ego_lane_lw  = lw_base
        
        #
        longitudinalforce = .0
        lateralforce = .0
        #######################################LEFT LANE
        for vid in vehs_info_within_area_left.keys():
            #
            #============================total force
            neighbor_state = vehs_info_within_area_left[vid]
            #
            deltax = ego_state[0] - neighbor_state[0]
            deltay = ego_state[2] - neighbor_state[2]
            #deltay = deltay_not_scaled*2
            #totalforce = force_scale_deletable*abs(tmp1 - tmp2)
            tmp0  = -(ego_state[1]**2)/2.0*abs(deltax)
            #   tmp1 must be positive, as abs(deltay) must be smaller than 
            tmp1 = 1.0/(abs(deltay) - ego_lane_lw/2.0)
            #   tmp2 MUST BE positive. 
            tmp2 = 1.0/(ellipse_y_left*np.sqrt(1.0 - (deltax**2)/(ellipse_x**2)) - ego_lane_lw/2.0)
            #
            #--------------------total force
            #totalforce = abs(tmp0*(tmp1 - tmp2))
            totalforce = force_scale_deletable*abs(tmp1 - tmp2)
            
            #
            #====================================longitudinalforce, MUST BE NEGATIVE. 
            #print(deltax, deltay)
            longitudinalforce = longitudinalforce - totalforce*abs(deltax)/(np.sqrt(deltax**2 + deltay**2))
            #--------------------------------------
            lateralforce = lateralforce - totalforce*abs(deltay)/(np.sqrt(deltax**2 + deltay**2))
            """
            if ego_state[2] > neighbor_state[2]:
                lateralforce = lateralforce + totalforce*abs(deltay)/(np.sqrt(deltax**2 + deltay**2))
            else:
                lateralforce = lateralforce -totalforce*abs(deltay)/(np.sqrt(deltax**2 + deltay**2))
            """
        #######################################RIGHT LANE
        for vid in vehs_info_within_area_right.keys():
            #
            #============================total force
            neighbor_state = vehs_info_within_area_right[vid]
            #
            deltax = ego_state[0] - neighbor_state[0]
            deltay = ego_state[2] - neighbor_state[2]
            #totalforce
            tmp0  = -(ego_state[1]**2)/2.0*abs(deltax)
            #   tmp1 must be positive, as abs(deltay) must be smaller than 
            tmp1 = 1.0/(abs(deltay) - ego_lane_lw/2.0)
            #   tmp2 MUST BE positive. 
            tmp2 = 1.0/(ellipse_y_right*np.sqrt(1.0 - (deltax**2)/(ellipse_x**2)) - ego_lane_lw/2.0)
            #
            #--------------------total force
            totalforce = abs(tmp0*(tmp1 - tmp2))
            #totalforce = force_scale_deletable*abs(tmp1 - tmp2)
            #
            #====================================longitudinalforce, MUST BE NEGATIVE. 
            longitudinalforce = longitudinalforce - totalforce*abs(deltax)/(np.sqrt(deltax**2 + deltay**2))
            #
            lateralforce = lateralforce + totalforce*abs(deltay)/(np.sqrt(deltax**2 + deltay**2))
            #--------------------------------------
            """
            if ego_state[2] > neighbor_state[2]:
                lateralforce = lateralforce + totalforce*abs(deltay)/(np.sqrt(deltax**2 + deltay**2))
            else:
                lateralforce = lateralforce -totalforce*abs(deltay)/(np.sqrt(deltax**2 + deltay**2))
            """
        
        return longitudinalforce,lateralforce



    @classmethod
    def PotentialNeighboringLongitudinalLateral(self, ego_state, ego_lane_lw, \
            vehs_info_within_area_left, \
            vehs_info_within_area_right, \
            ellipse_x, ellipse_y_left, ellipse_y_right, \
            force_scale_deletable = 1.0):
        """
        Callback
        
            longitudinalforce,lateralforce = self.PotentialNeighboringLongitudinalLateral(ego_state, ego_lane_lw, vehs_ids_within_area, ellipse_x, ellipse_y)
        
        ---------------------------------------------------------------
        @input: ego_state
        
            ego_state = np.array([x, vx, y, vy, zlon, zlat])
        
        @input: vehs_ids_within_area
            
            the keys are vids. 
            
            vehs_ids_within_area[vid] = np.array([x, vx, y, vy, zlon, zlat])
            
            vehs_ids_within_area = {}
        
        @OUTPUT: longitudinalforce,lateralforce
        
            two floats. 
            
            The sign indicate the direction of the force. 
            
            negative means that the force points to the negative direction of the y axis. 
        
        --------------------------------------------------------
        
        """
        longitudinalforce = .0
        lateralforce = .0
        #######################################LEFT LANE
        for vid in vehs_info_within_area_left.keys():
            #
            #============================total force
            neighbor_state = vehs_info_within_area_left[vid]
            #
            deltax = ego_state[0] - neighbor_state[0]
            deltay = ego_state[2] - neighbor_state[2]
            #deltay = deltay_not_scaled*2
            #totalforce = force_scale_deletable*abs(tmp1 - tmp2)
            tmp0  = -(ego_state[1]**2)/2.0*abs(deltax)
            #   tmp1 must be positive, as abs(deltay) must be smaller than 
            tmp1 = 1.0/(abs(deltay) - ego_lane_lw/2.0)
            #   tmp2 MUST BE positive. 
            tmp2 = 1.0/(ellipse_y_left*np.sqrt(1.0 - (deltax**2)/(ellipse_x**2)) - ego_lane_lw/2.0)
            #
            #--------------------total force
            #totalforce = abs(tmp0*(tmp1 - tmp2))
            totalforce = force_scale_deletable*abs(tmp1 - tmp2)
            
            #
            #====================================longitudinalforce, MUST BE NEGATIVE. 
            #print(deltax, deltay)
            longitudinalforce = longitudinalforce - totalforce*abs(deltax)/(np.sqrt(deltax**2 + deltay**2))
            #--------------------------------------
            lateralforce = lateralforce - totalforce*abs(deltay)/(np.sqrt(deltax**2 + deltay**2))
            """
            if ego_state[2] > neighbor_state[2]:
                lateralforce = lateralforce + totalforce*abs(deltay)/(np.sqrt(deltax**2 + deltay**2))
            else:
                lateralforce = lateralforce -totalforce*abs(deltay)/(np.sqrt(deltax**2 + deltay**2))
            """
        #######################################RIGHT LANE
        for vid in vehs_info_within_area_right.keys():
            #
            #============================total force
            neighbor_state = vehs_info_within_area_right[vid]
            #
            deltax = ego_state[0] - neighbor_state[0]
            deltay = ego_state[2] - neighbor_state[2]
            #totalforce
            tmp0  = -(ego_state[1]**2)/2.0*abs(deltax)
            #   tmp1 must be positive, as abs(deltay) must be smaller than 
            tmp1 = 1.0/(abs(deltay) - ego_lane_lw/2.0)
            #   tmp2 MUST BE positive. 
            tmp2 = 1.0/(ellipse_y_right*np.sqrt(1.0 - (deltax**2)/(ellipse_x**2)) - ego_lane_lw/2.0)
            #
            #--------------------total force
            totalforce = abs(tmp0*(tmp1 - tmp2))
            #totalforce = force_scale_deletable*abs(tmp1 - tmp2)
            #
            #====================================longitudinalforce, MUST BE NEGATIVE. 
            longitudinalforce = longitudinalforce - totalforce*abs(deltax)/(np.sqrt(deltax**2 + deltay**2))
            #
            lateralforce = lateralforce + totalforce*abs(deltay)/(np.sqrt(deltax**2 + deltay**2))
            #--------------------------------------
            """
            if ego_state[2] > neighbor_state[2]:
                lateralforce = lateralforce + totalforce*abs(deltay)/(np.sqrt(deltax**2 + deltay**2))
            else:
                lateralforce = lateralforce -totalforce*abs(deltay)/(np.sqrt(deltax**2 + deltay**2))
            """
        
        return longitudinalforce,lateralforce





    @classmethod
    def PotentialNeighboringLateral(self, ego_state, vehs_ids_within_area):
        """
        The lateral component of the vehicles from neighboring lanes. 
        
        --------------------------------------------------------
        @input: ego_state
        
            ego_state = np.array([x, vx, y, vy, zlon, zlat])
        
        @input: vehs_ids_within_area
            
            the keys are vids. 
            
            vehs_ids_within_area[vid] = np.array([x, vx, y, vy, zlon, zlat])
            
            vehs_ids_within_area = {}
        
        @OUTPUT: lateralforce
        
            a float. 
            
            The sign indicate the direction of the force. 
            
            negative means that the force points to the negative direction of the y axis. 
        
        --------------------------------------------------------
        
        """
        
        
        
        pass
    

    @classmethod
    def PotentialNeighboringLongitudinal(self, ego_state, vehs_info_within_area):
        """
        The lateral component of the vehicles from neighboring lanes. 
        
        -------------------------------------------------
        @input: ego_state
        
            ego_state = np.array([x, vx, y, vy, zlon, zlat])
        
        @vehs_info_within_area
        
            a dict. keys are the vehicle ids and value are the vehicle states. 
        
        
        @OUTPUT: longitudinalforces
        
            a dict.  longitudinalforces[vid] is a float. 
            
           It should be always negative. 
        
        
        """
        
        
        for vid in vehs_info_within_area.keys():
            #
            neighbor_state = vehs_info_within_area[vid]
            #
            deltax = ego_state[0] - neighbor_state[0]
            deltay = ego_state[2] - neighbor_state[2]
            #totalforce
            tmp0  = -(ego_state[0]**2)/2.0*deltax
            #tmp1 = 1.0/(deltay - )
            
            
            
            pass
        
        
        longitudinalforces = {}
        
        
        
        
        
        
        
        pass

    @classmethod
    def PotentialLaneMark(self, ego_state, ego_lane_marks_coor, two_dim_paras = two_dim_paras, right_force_exist = True, left_force_exist = True):
        """
        
        -----------------------------------------------
        
        @input: ego_state
        
            ego_state
        
            a  state of the ego vehicle. 
            
            
            The state of the vehicle is represented as:
        
            - np.array([x, vx, y, vy, zlon, zlat])  x is the longitudinal, y is the lateral, z is the noise. 
        
        @input: two_dim_paras
        
            the parameters for the two dimensional microscopic model. 
        
        @input: ego_lane_marks_coor
        
            ego_lane_marks_coor = (float1, float2)
        
        @input: lane_middle_line_ccor
            
            a float which represent the lane middle line coordinate. 
        
        -------------------------------------------------------
        @OUTPUT: potential
        
            a float which represent the force generated by the lane mark. 
            NOTE THAT THE sign. 
            
            
        """
        
        #
        x,vx,y, vy,Z_long,Z_lat = ego_state[0],ego_state[1],ego_state[2],ego_state[3],ego_state[4],ego_state[5]
        #
        beta_lane_marks = two_dim_paras['beta_lane_marks']
        lw = abs(min(ego_lane_marks_coor) - max(ego_lane_marks_coor))
        #
        potential = 0
        for markcoor in ego_lane_marks_coor:
            deltay = y - markcoor
            #y>markcoor means this mark is at right hand side.
            if y>markcoor:
                #print(deltay)
                if right_force_exist and abs(deltay)<lw/2.0:
                    potential = potential + (np.exp(-beta_lane_marks*deltay*deltay) - np.exp(-beta_lane_marks*lw*lw/4.0))
                    #potential = potential + (np.exp(-beta_lane_marks*deltay*deltay))
                
            else:
                if left_force_exist and abs(deltay)<lw/2.0:
                    #potential = potential - (np.exp(-beta_lane_marks*deltay*deltay))
                    potential = potential - (np.exp(-beta_lane_marks*deltay*deltay) - np.exp(-beta_lane_marks*lw*lw/4.0))
            
        return potential


    @classmethod
    def PotentialLaneMark_parabolic(self, ego_state, ego_lane_marks_coor, two_dim_paras = two_dim_paras, right_force_exist = True, left_force_exist = True):
        """
        Difference: 
            - self.PotentialLaneMark_parabolic()
            - self.PotentialLaneMark()
        
        -----------------------------------------
        @input: ego_state
        
            ego_state
        
            a  state of the ego vehicle. 
            
            
            The state of the vehicle is represented as:
        
            - np.array([x, vx, y, vy, zlon, zlat])  x is the longitudinal, y is the lateral, z is the noise. 
        
        @input: two_dim_paras
        
            the parameters for the two dimensional microscopic model. 
        
        @input: ego_lane_marks_coor
        
            ego_lane_marks_coor = (float1, float2)
        
        @input: lane_middle_line_ccor
            
            a float which represent the lane middle line coordinate. 
        
        -------------------------------------------------------
        @OUTPUT: potential
        
            a float which represent the force generated by the lane mark. 
            NOTE THAT THE sign. 
            
            
        """
        
        #
        x,vx,y, vy,Z_long,Z_lat = ego_state[0],ego_state[1],ego_state[2],ego_state[3],ego_state[4],ego_state[5]
        #
        beta_lane_marks = two_dim_paras['beta_lane_marks']
        lw = abs(min(ego_lane_marks_coor) - max(ego_lane_marks_coor))
        #
        potential = 0
        for markcoor in ego_lane_marks_coor:
            deltay = y - markcoor
            #y>markcoor means this mark is at right hand side. Then the force must be positive. 
            if y>markcoor:
                #print(deltay)
                if right_force_exist and deltay<lw/2.0:# and abs(deltay)<lw/2.0:
                    potential = potential + (deltay - lw/2.0)**2
                    #potential = potential + (np.exp(-beta_lane_marks*deltay*deltay) - np.exp(-beta_lane_marks*lw*lw/4.0))
                    #potential = potential + (np.exp(-beta_lane_marks*deltay*deltay))
                
            else:
                if left_force_exist and deltay>-lw/2.0:# and abs(deltay)<lw/2.0:
                    potential = potential - (deltay - lw/2.0)**2
                    #potential = potential - (np.exp(-beta_lane_marks*deltay*deltay))
                    #potential = potential - (np.exp(-beta_lane_marks*deltay*deltay) - np.exp(-beta_lane_marks*lw*lw/4.0))
            
        return potential
        ###############################################################################################
        ###############################################################################################
        ###############################################################################################
        #
        #x,vx,y, vy,Z_long,Z_lat = ego_state[0],ego_state[1],ego_state[2],ego_state[3],ego_state[4],ego_state[5]
        #
        beta_lane_marks = two_dim_paras['beta_lane_marks']
        #
        potential = 0
        for markcoor in lane_mark_coors:
            deltay = y - markcoor
            if y>markcoor:
                #print(deltay)
                if right_force_exist and abs(deltay)<lw/2.0:
                    potential = potential + np.exp(-beta_lane_marks*deltay*deltay)
                    
                
            else:
                if left_force_exist and abs(deltay)<lw/2.0:
                    #
                    potential = potential - np.exp(-beta_lane_marks*deltay*deltay)
            
        return potential
        
        

    @classmethod
    def F(self, ego_state, \
            vehs_info_within_area_left, \
            vehs_info_within_area_right, \
            ego_lane_lw, \
            ego_lane_middle_line_coor, \
            ego_lane_marks_coor, \
            road_bounds, \
            lw_boundary_lanes, \
            leader_state, \
            ellipse_x, ellipse_y_left, ellipse_y_right, \
            two_dim_paras = two_dim_paras, \
            idm_paras = idm_paras, \
            stochastic_proecess_name = 'OU', \
            with_neighboring_influence = True, \
            right_mark_force = True,  \
            left_bound_force = True, \
            left_mark_force = True, 
            right_bound_force = True, \
            lanemark_force_type = 'parabolic' ):
        """
        dS = F(S) dt + L(S) dW
        -----------------------------------------
        
        @input: lanemark_force_type
        
            either 'parabolic' or 'exponential'
        
        
        @input:     
            
            The state of the vehicle is represented as:
        
            - x,vx,y, vy,Z_long,Z_lat = ego_state[0],ego_state[1],ego_state[2],ego_state[3],ego_state[4],ego_state[5]
            
        @Input: neighbores_states_dict
        
            a dict containing the state of the neighbores of the ego vehicles. 
        
        @input: with_neighboring_influence
        
            whether the neighboring lanes influence should be accounted. 
        
        @input: vehs_target_lateral_dict
        
            a dict. keys are vids and values are the lane middle line coordinate at target lane. 
            
        ----------------------------------------------
        @OUTPUT: np.array([diff_x, diff_vx, diff_y, diff_vy, diff_Z_long, diff_Z_lat])
        
        
        """
        sigma_long_drift = two_dim_paras['sigma_long_drift']
        sigma_lat_drift = two_dim_paras['sigma_lat_drift']
        sigma_long = two_dim_paras['sigma_long']
        sigma_lat = two_dim_paras['sigma_lat']
        
        #
        x,vx,y,vy,Z_long,Z_lat = ego_state[0],ego_state[1],ego_state[2],ego_state[3],ego_state[4],ego_state[5]
        
        #==============================
        diff_x = vx
        diff_y = vy

        #==============================diff_Z_long, diff_Z_lat
        if stochastic_proecess_name=='OU':
            #diff_Z_long = -(sigma_long**1.0)*(Z_long**3)
            #diff_Z_lat = -(sigma_lat**1.0)*(Z_lat)
            #print(Z_long, sigma_long_drift)
            diff_Z_long = -sigma_long_drift*Z_long
            diff_Z_lat = -sigma_lat_drift*Z_lat
        elif stochastic_proecess_name=='simple':
            #diff_Z_long = -(sigma_long**1.0)*(Z_long**3)
            #diff_Z_lat = -(sigma_lat**1.0)*(Z_lat)
            #print(Z_long, sigma_long_drift)
            diff_Z_long = 0
            diff_Z_lat = 0
        elif stochastic_proecess_name=='converted':
            #the converted. 
            diff_Z_long = -(sigma_long**2)*(1-Z_long**2)*Z_long
            diff_Z_lat = -(sigma_lat**2)*(1-Z_lat**2)*Z_lat
        elif stochastic_proecess_name=='geometric':
            #
            diff_Z_long =  -sigma_long_drift*(Z_long)
            diff_Z_lat = -sigma_lat_drift*(Z_lat)
            #
        elif stochastic_proecess_name=='jacobi':
            #
            diff_Z_long = -sigma_long_drift*(Z_long - .0)
            diff_Z_lat = -sigma_lat_drift*(Z_lat - .0)
            #
        elif stochastic_proecess_name=='hyperparabolic':
            #
            #diff_Z_long = -sigma_long_drift*(Z_long - .0)
            #diff_Z_lat = -sigma_lat_drift*(Z_lat - .0)
            diff_Z_long =  -Z_long-sigma_long_drift*Z_long
            diff_Z_lat = -Z_lat-sigma_lat_drift*Z_lat
            #print(diff_Z_long, diff_Z_lat)
        elif stochastic_proecess_name=='ROU':
            #ew_state = STATES[-1] + (theta/STATES[-1] -  STATES[-1] )*deltat + sigma*brownian
            diff_Z_long = -sigma_long_drift/Z_long +  Z_long#-sigma_long_drift*(Z_long - .0)
            diff_Z_lat = -sigma_lat_drift/Z_lat +  Z_lat #-sigma_lat_drift*(Z_lat - .0)
        #
        #==============================diff_vx and diff_vy
        #potentials is a dict. Its keys are 'boundary', 'idm', 'lanemark', 'neighboringlongitudial', 'neighboringlateral', 'lanemiddleline'
        potentials = self.Potential(ego_state = ego_state, \
            vehs_info_within_area_left = vehs_info_within_area_left, \
            vehs_info_within_area_right = vehs_info_within_area_right, \
            ego_lane_lw = ego_lane_lw, \
            ego_lane_middle_line_coor = ego_lane_middle_line_coor, \
            ego_lane_marks_coor = ego_lane_marks_coor, \
            road_bounds = road_bounds, \
            lw_boundary_lanes = lw_boundary_lanes, \
            leader_state = leader_state, \
            ellipse_x = ellipse_x, \
            ellipse_y_left = ellipse_y_left, \
            ellipse_y_right = ellipse_y_right, \
            two_dim_paras = two_dim_paras,\
            idm_paras = idm_paras, \
            right_mark_force = right_mark_force,  \
            left_bound_force = left_bound_force, \
            left_mark_force = left_mark_force, 
            right_bound_force = right_bound_force, \
            lanemark_force_type  = lanemark_force_type)
        #
        #--------------diff_vx
        if with_neighboring_influence:
            diff_vx = diff_Z_long + potentials['idm'] + \
                two_dim_paras['amplyfier_intra_lanes_long']*potentials['neighboringlongitudinal']
        else:
            diff_vx = diff_Z_long + potentials['idm']# + potentials['neighboringlongitudinal']
        #---------------diff_vy
        if with_neighboring_influence:
            diff_vy = diff_Z_lat + \
                potentials['lanemiddleline'] + \
                two_dim_paras['amplyfier_bound']*potentials['boundary']   + \
                two_dim_paras['amplyfier_lane_mark']*potentials['lanemark'] + \
                two_dim_paras['amplyfier_intra_lanes_lat']*potentials['neighboringlateral']
        else:
            diff_vy = diff_Z_lat + \
                potentials['lanemiddleline'] + \
                two_dim_paras['amplyfier_bound']*potentials['boundary']   + \
                two_dim_paras['amplyfier_lane_mark']*potentials['lanemark']
        #
        #diff_vx = potentials['idm'] + potentials['neighboringlongitudinal'] + diff_Z_long
        #diff_vy = potentials['neighboringlateral'] + potentials['boundary'] + potentials['lanemark'] + potentials['lanemiddleline'] + diff_Z_lat
        #
        #
        return np.array([diff_x, diff_vx, diff_y, diff_vy, diff_Z_long, diff_Z_lat]),potentials


    @classmethod
    def F_lw_base(self, ego_state, \
            vehs_info_within_area_left, \
            vehs_info_within_area_right, \
            ego_lane_lw, \
            ego_lane_middle_line_coor, \
            ego_lane_marks_coor, \
            road_bounds, \
            lw_boundary_lanes, \
            leader_state, \
            ellipse_x, ellipse_y_left, ellipse_y_right, \
            two_dim_paras = two_dim_paras, \
            idm_paras = idm_paras, \
            stochastic_proecess_name = 'OU', \
            with_neighboring_influence = True, \
            right_mark_force = True,  \
            left_bound_force = True, \
            left_mark_force = True, 
            right_bound_force = True, \
            lanemark_force_type = 'parabolic', lw_base = 3.5 ):
        """
        Difference between:
        
            - self.F()
            - self.F_lw_base()
        
        The latter is for the case of lane width, 
        
        
        dS = F(S) dt + L(S) dW
        -----------------------------------------
        
        @input: lanemark_force_type
        
            either 'parabolic' or 'exponential'
        
        
        @input:     
            
            The state of the vehicle is represented as:
        
            - x,vx,y, vy,Z_long,Z_lat = ego_state[0],ego_state[1],ego_state[2],ego_state[3],ego_state[4],ego_state[5]
            
        @Input: neighbores_states_dict
        
            a dict containing the state of the neighbores of the ego vehicles. 
        
        @input: with_neighboring_influence
        
            whether the neighboring lanes influence should be accounted. 
        
        @input: vehs_target_lateral_dict
        
            a dict. keys are vids and values are the lane middle line coordinate at target lane. 
            
        ----------------------------------------------
        @OUTPUT: np.array([diff_x, diff_vx, diff_y, diff_vy, diff_Z_long, diff_Z_lat])
        
        
        """
        sigma_long_drift = two_dim_paras['sigma_long_drift']
        sigma_lat_drift = two_dim_paras['sigma_lat_drift']
        sigma_long = two_dim_paras['sigma_long']
        sigma_lat = two_dim_paras['sigma_lat']
        
        #
        x,vx,y,vy,Z_long,Z_lat = ego_state[0],ego_state[1],ego_state[2],ego_state[3],ego_state[4],ego_state[5]
        
        #==============================
        diff_x = vx
        diff_y = vy

        #==============================diff_Z_long, diff_Z_lat
        if stochastic_proecess_name=='OU':
            #diff_Z_long = -(sigma_long**1.0)*(Z_long**3)
            #diff_Z_lat = -(sigma_lat**1.0)*(Z_lat)
            #print(Z_long, sigma_long_drift)
            diff_Z_long = -sigma_long_drift*Z_long
            diff_Z_lat = -sigma_lat_drift*Z_lat
        elif stochastic_proecess_name=='simple':
            #diff_Z_long = -(sigma_long**1.0)*(Z_long**3)
            #diff_Z_lat = -(sigma_lat**1.0)*(Z_lat)
            #print(Z_long, sigma_long_drift)
            diff_Z_long = 0
            diff_Z_lat = 0
        elif stochastic_proecess_name=='converted':
            #the converted. 
            diff_Z_long = -(sigma_long**2)*(1-Z_long**2)*Z_long
            diff_Z_lat = -(sigma_lat**2)*(1-Z_lat**2)*Z_lat
        elif stochastic_proecess_name=='geometric':
            #
            diff_Z_long =  -sigma_long_drift*(Z_long)
            diff_Z_lat = -sigma_lat_drift*(Z_lat)
            #
        elif stochastic_proecess_name=='jacobi':
            #
            diff_Z_long = -sigma_long_drift*(Z_long - .0)
            diff_Z_lat = -sigma_lat_drift*(Z_lat - .0)
            #
        elif stochastic_proecess_name=='hyperparabolic':
            #
            #diff_Z_long = -sigma_long_drift*(Z_long - .0)
            #diff_Z_lat = -sigma_lat_drift*(Z_lat - .0)
            diff_Z_long =  -Z_long-sigma_long_drift*Z_long
            diff_Z_lat = -Z_lat-sigma_lat_drift*Z_lat
            #print(diff_Z_long, diff_Z_lat)
        elif stochastic_proecess_name=='ROU':
            #ew_state = STATES[-1] + (theta/STATES[-1] -  STATES[-1] )*deltat + sigma*brownian
            diff_Z_long = -sigma_long_drift/Z_long +  Z_long#-sigma_long_drift*(Z_long - .0)
            diff_Z_lat = -sigma_lat_drift/Z_lat +  Z_lat #-sigma_lat_drift*(Z_lat - .0)
        #
        #==============================diff_vx and diff_vy
        #potentials is a dict. Its keys are 'boundary', 'idm', 'lanemark', 'neighboringlongitudial', 'neighboringlateral', 'lanemiddleline'
        potentials = self.Potential_lw_base(ego_state = ego_state, \
            vehs_info_within_area_left = vehs_info_within_area_left, \
            vehs_info_within_area_right = vehs_info_within_area_right, \
            ego_lane_lw = ego_lane_lw, \
            ego_lane_middle_line_coor = ego_lane_middle_line_coor, \
            ego_lane_marks_coor = ego_lane_marks_coor, \
            road_bounds = road_bounds, \
            lw_boundary_lanes = lw_boundary_lanes, \
            leader_state = leader_state, \
            ellipse_x = ellipse_x, \
            ellipse_y_left = ellipse_y_left, \
            ellipse_y_right = ellipse_y_right, \
            two_dim_paras = two_dim_paras,\
            idm_paras = idm_paras, \
            right_mark_force = right_mark_force,  \
            left_bound_force = left_bound_force, \
            left_mark_force = left_mark_force, 
            right_bound_force = right_bound_force, \
            lanemark_force_type  = lanemark_force_type, lw_base = lw_base)
        #
        #--------------diff_vx
        if with_neighboring_influence:
            diff_vx = diff_Z_long + potentials['idm'] + \
                two_dim_paras['amplyfier_intra_lanes_long']*potentials['neighboringlongitudinal']
        else:
            diff_vx = diff_Z_long + potentials['idm']# + potentials['neighboringlongitudinal']
        #---------------diff_vy
        if with_neighboring_influence:
            diff_vy = diff_Z_lat + \
                potentials['lanemiddleline'] + \
                two_dim_paras['amplyfier_bound']*potentials['boundary']   + \
                two_dim_paras['amplyfier_lane_mark']*potentials['lanemark'] + \
                two_dim_paras['amplyfier_intra_lanes_lat']*potentials['neighboringlateral']
        else:
            diff_vy = diff_Z_lat + \
                potentials['lanemiddleline'] + \
                two_dim_paras['amplyfier_bound']*potentials['boundary']   + \
                two_dim_paras['amplyfier_lane_mark']*potentials['lanemark']
        #
        #diff_vx = potentials['idm'] + potentials['neighboringlongitudinal'] + diff_Z_long
        #diff_vy = potentials['neighboringlateral'] + potentials['boundary'] + potentials['lanemark'] + potentials['lanemiddleline'] + diff_Z_lat
        #
        #
        return np.array([diff_x, diff_vx, diff_y, diff_vy, diff_Z_long, diff_Z_lat]),potentials
    
    @classmethod
    def IDM_equilibrium_v_from_deltax(self, deltax, idm_paras = idm_paras):
        """
        Given the space gap, find the speed. 
        
        deltax unit is m
        
        returned speed unit is m/s
        
        """
        #
        idm_vf = idm_paras['idm_vf']
        idm_T = idm_paras['idm_T']
        idm_delta = idm_paras['idm_delta']
        idm_s0 = idm_paras['idm_s0']
        idm_a = idm_paras['idm_a']
        idm_b = idm_paras['idm_b']
        veh_len = idm_paras['veh_len']
        
        #k unit is veh/km.
        k = 1000.0/deltax
        kjam = 1000.0/(idm_s0 + veh_len)
        #
        return max(0, idm_vf - idm_vf/kjam*k)
    
    @classmethod
    def ttc(self, v_leader_ms = 10, v_follower_ms = 8, deltax_m = 10, M = 1e10):
        """
        calculate the time to collision. 
        
        
        """
        if v_leader_ms>=v_follower_ms:
            return M
        #
        #ttc*v_leader_ms + deltax_m = ttc*v_follower_ms
        return deltax_m/(v_follower_ms - v_leader_ms)
        #
        
    @classmethod
    def BackwardInfluenceArea(self, state_ego):
        """
        
        """
        
        
        
        
        pass


    @classmethod
    def Speed_upperlower_given_jerk_upperlower_lat(self, v0, v1, deltat0, deltat1, ):
        """
        Difference:
        
            self.Speed_upperlower_given_jerk_upperlower
            self.Speed_upperlower_given_jerk_upperlower_lat
            
        The latter one is for lat, using self.SpeedDependentJumpAdjust_lat, rather than SpeedDependentJumpAdjust. 
        
        Calculate the upper and lower of the speed, consideing that the  
        
        Callback: 
        
            minn,maxx = TwoDimMicroModel.Speed_upperlower_given_jerk_upperlower(v0, v1, deltat0, delta1, )
        
        
        The jerk is calculated as: ((v2 - v1)/deltat1 - (v1 - v0)/deltat0)/deltat1 . 
        
        IT is within [jerk_min, jerk_max], and thus v2, the new speed, need to satisfy:
        
            v2 \in [v2_min, v2_max]
        Derivation:
            
            The maximum component: 
            
                ((v2 - v1)/deltat1 - (v1 - v0)/deltat0)/deltat1 < jerk_max
                
                (v2 - v1)/deltat1  < jerk_max*deltat1 + (v1 - v0)/deltat0
                
                (v2 - v1) < (jerk_max*deltat1 + (v1 - v0)/deltat0)*deltat1
                
                v2 < v1 + (jerk_max*deltat1 + (v1 - v0)/deltat0)*deltat1
            
            The minimum component:
            
                ((v2 - v1)/deltat1 - (v1 - v0)/deltat0)/deltat1 > jerk_min
                
                ((v2 - v1)/deltat1  > jerk_min*deltat1 + (v1 - v0)/deltat0
                
                (v2 - v1) > (jerk_min*deltat1 + (v1 - v0)/deltat0)*deltat1
                
                v2 > v1 + (jerk_min*deltat1 + (v1 - v0)/deltat0)*deltat1
            
        ---------------------------------------------------------------
        @input: v0, v1
        
            both in m/s
            
        @input: deltat0, deltat1
        
            both in second 
        
        @output: minn,maxx
        
            
        
        
        
        """
        #max and min jerk
        jerk_max = self.SpeedDependentJumpAdjust_lat(v1)
        jerk_min = -jerk_max
        #
        #
        maxx = v1 + (jerk_max*deltat1 + (v1 - v0)/deltat0)*deltat1
        minn = v1 + (jerk_min*deltat1 + (v1 - v0)/deltat0)*deltat1
        
        return minn,maxx


    @classmethod
    def Speed_upperlower_given_jerk_upperlower(self, v0, v1, deltat0, deltat1, ):
        """
        Calculate the upper and lower of the speed, consideing that the  
        
        Callback: 
        
            minn,maxx = TwoDimMicroModel.Speed_upperlower_given_jerk_upperlower(v0, v1, deltat0, delta1, )
        
        
        The jerk is calculated as: ((v2 - v1)/deltat1 - (v1 - v0)/deltat0)/deltat1 . 
        
        IT is within [jerk_min, jerk_max], and thus v2, the new speed, need to satisfy:
        
            v2 \in [v2_min, v2_max]
        Derivation:
            
            The maximum component: 
            
                ((v2 - v1)/deltat1 - (v1 - v0)/deltat0)/deltat1 < jerk_max
                
                (v2 - v1)/deltat1  < jerk_max*deltat1 + (v1 - v0)/deltat0
                
                (v2 - v1) < (jerk_max*deltat1 + (v1 - v0)/deltat0)*deltat1
                
                v2 < v1 + (jerk_max*deltat1 + (v1 - v0)/deltat0)*deltat1
            
            The minimum component:
            
                ((v2 - v1)/deltat1 - (v1 - v0)/deltat0)/deltat1 > jerk_min
                
                ((v2 - v1)/deltat1  > jerk_min*deltat1 + (v1 - v0)/deltat0
                
                (v2 - v1) > (jerk_min*deltat1 + (v1 - v0)/deltat0)*deltat1
                
                v2 > v1 + (jerk_min*deltat1 + (v1 - v0)/deltat0)*deltat1
            
        ---------------------------------------------------------------
        @input: v0, v1
        
            both in m/s
            
        @input: deltat0, deltat1
        
            both in second 
        
        @output: minn,maxx
        
            
        
        
        
        """
        #max and min jerk
        jerk_max = self.SpeedDependentJumpAdjust(v1)
        jerk_min = -jerk_max
        #
        #
        maxx = v1 + (jerk_max*deltat1 + (v1 - v0)/deltat0)*deltat1
        minn = v1 + (jerk_min*deltat1 + (v1 - v0)/deltat0)*deltat1
        
        return minn,maxx
    
    @classmethod
    def F_SingleLane(self, ego_state, \
            ego_lane_lw, \
            ego_lane_middle_line_coor, \
            ego_lane_marks_coor, \
            road_bounds, \
            lw_boundary_lanes, \
            leader_state, \
            two_dim_paras = two_dim_paras, \
            stochastic_proecess_name = 'OU', \
            deltax_2d_or_not = False):
        """
        
        -----------------------------------------
        dS = F(S) dt + L(S) dW
        
        @input:     
            
            The state of the vehicle is represented as:
        
            - x,vx,y, vy,Z_long,Z_lat = ego_state[0],ego_state[1],ego_state[2],ego_state[3],ego_state[4],ego_state[5]
        
        @input: deltax_2d_or_not
            
            when calculateth the acceleration for the followers, the delatax in the IDM model can be set as the longitudinal distance or the 2d distance. 
            
            if yes, then the deltax is the 2d distance
            it not then the deladx is the conventional 1d. 
        
        @Input: neighbores_states_dict
        
            a dict containing the state of the neighbores of the ego vehicles. 
        
        @input: vehs_target_lateral_dict
        
            a dict. keys are vids and values are the lane middle line coordinate at target lane. 
            
        ----------------------------------------------
        @OUTPUT: np.array([diff_x, diff_vx, diff_y, diff_vy, diff_Z_long, diff_Z_lat])
        
        
        """
        sigma_long_drift = two_dim_paras['sigma_long_drift']
        sigma_lat_drift = two_dim_paras['sigma_lat_drift']
        sigma_long = two_dim_paras['sigma_long']
        sigma_lat = two_dim_paras['sigma_lat']
        #
        x,vx,y,vy,Z_long,Z_lat = ego_state[0],ego_state[1],ego_state[2],ego_state[3],ego_state[4],ego_state[5]
        
        #==============================
        diff_x = vx
        diff_y = vy

        #==============================diff_Z_long, diff_Z_lat
        if stochastic_proecess_name=='OU':
            #diff_Z_long = -(sigma_long**1.0)*(Z_long**3)
            #diff_Z_lat = -(sigma_lat**1.0)*(Z_lat)
            #print(Z_long, sigma_long_drift)
            diff_Z_long = -sigma_long_drift*Z_long
            diff_Z_lat = -sigma_lat_drift*Z_lat
        elif stochastic_proecess_name=='simple':
            #diff_Z_long = -(sigma_long**1.0)*(Z_long**3)
            #diff_Z_lat = -(sigma_lat**1.0)*(Z_lat)
            #print(Z_long, sigma_long_drift)
            diff_Z_long = 0
            diff_Z_lat = 0
        elif stochastic_proecess_name=='converted':
            #the converted. 
            diff_Z_long = -(sigma_long**2)*(1-Z_long**2)*Z_long
            diff_Z_lat = -(sigma_lat**2)*(1-Z_lat**2)*Z_lat
        elif stochastic_proecess_name=='geometric':
            #
            diff_Z_long =  -sigma_long_drift*(Z_long)
            diff_Z_lat = -sigma_lat_drift*(Z_lat)
            #
        elif stochastic_proecess_name=='jacobi':
            #
            diff_Z_long = -sigma_long_drift*(Z_long - .0)
            diff_Z_lat = -sigma_lat_drift*(Z_lat - .0)
            #
        elif stochastic_proecess_name=='hyperparabolic':
            #
            #diff_Z_long = -sigma_long_drift*(Z_long - .0)
            #diff_Z_lat = -sigma_lat_drift*(Z_lat - .0)
            diff_Z_long =  -Z_long-sigma_long_drift*Z_long
            diff_Z_lat = -Z_lat-sigma_lat_drift*Z_lat
            #print(diff_Z_long, diff_Z_lat)
        elif stochastic_proecess_name=='ROU':
            #ew_state = STATES[-1] + (theta/STATES[-1] -  STATES[-1] )*deltat + sigma*brownian
            diff_Z_long = -sigma_long_drift/Z_long +  Z_long#-sigma_long_drift*(Z_long - .0)
            diff_Z_lat = -sigma_lat_drift/Z_lat +  Z_lat #-sigma_lat_drift*(Z_lat - .0)
        
        #
        #==============================diff_vx and diff_vy
        #potentials is a dict. Its keys are 'boundary', 'idm', 'lanemark', 'neighboringlongitudial', 'neighboringlateral', 'lanemiddleline'
        potentials = {}
        #
        #------------------potentials['boundary']
        #PotentialRoadBoundary(ego_state, road_bounds, lw_boundary_lanes = [3.5, 3.5], two_dim_paras = two_dim_paras, right_force_exist = True, left_force_exist = True)
        potentials['boundary'] = self.PotentialRoadBoundary(ego_state = ego_state, road_bounds = road_bounds, lw_boundary_lanes = lw_boundary_lanes, two_dim_paras = two_dim_paras)
        #
        #-----------------potentials['idm']
        if deltax_2d_or_not:
            deltax = np.sqrt((leader_state[0] - ego_state[0])**2 + (leader_state[2] - ego_state[2])**2)
        else:
            deltax = abs(leader_state[0] - ego_state[0])
        #
        potentials['idm'] = self.Potential_IDM(v_self= ego_state[1]*3.6, v_leader = leader_state[1]*3.6, deltax = deltax, idm_paras = idm_paras)
        #
        #
        #-------------------potentials['lanemiddleline']
        potentials['lanemiddleline'] = self.PotentialLaneMiddleLine(ego_state  = ego_state, ego_lane_lw = ego_lane_lw, lane_middle_line_coor = ego_lane_middle_line_coor, two_dim_paras = two_dim_paras)
        #
        diff_vx = potentials['idm'] + diff_Z_long
        #diff_vx = potentials['idm'] + potentials['neighboringlongitudinal'] + diff_Z_long
        
        #
        diff_vy = potentials['lanemiddleline'] + \
                    two_dim_paras['amplyfier_bound']*potentials['boundary'] +  \
                    diff_Z_lat# + potentials['boundary']+ potentials['lanemiddleline']
        #diff_vy = potentials['lanemiddleline'] + diff_Z_lat# + potentials['boundary']+ potentials['lanemiddleline']
        #diff_vy = potentials['neighboringlateral'] + potentials['boundary'] + potentials['lanemark'] + potentials['lanemiddleline'] + diff_Z_lat
        
        #
        return np.array([diff_x, diff_vx, diff_y, diff_vy, diff_Z_long, diff_Z_lat]),potentials


    @classmethod
    def plot_Q_distribution_laneid_as_key(self, Q, ax = False,  figsize = (5,5), bins = 30):
        """
        Q,V,K = TwoDimMicroModel.Get_QVK_from_snapshots(self.snapshots, self.length)
        
            
            Q[laneid] = [q1, q2, q3, q4...]
            K[laneid] = [k1, k2, k3, k4...]
            V[laneid] = [v1,v2,....]
            Q unit is veh/h
            K unit is veh/km
        
        """
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)
            ax.set_xlabel('x');ax.set_ylabel('y'); 
        
        #
        for laneid in Q.keys():
            #
            hist0,edges = np.histogram(Q[laneid], bins = bins)
            #print(sum(hist0), edges[-1] - edges[-2])
            hist = hist0/sum(hist0)/(edges[-1] - edges[-2])
            ax.plot(edges[1:], hist, label = str(laneid))

        #
        ax.grid(0)
        #ax.legend()
        return ax
        
        #pass

    @classmethod
    def plot_QK_mean_laneid_as_key_im(self, QKVs_density_as_key, ax = False,  figsize = (5,5), color = 'b', alpha = .5, label = '2'):
        """
        
        QKVs_density_as_key[density] = Q,K,V

            Q[laneid] = [q1, q2, q3, q4...]
            K[laneid] = [k1, k2, k3, k4...]
            V[laneid] = [v1,v2,....]
            Q unit is veh/h
            K unit is veh/km
        
        """
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)
            ax.set_xlabel('x');ax.set_ylabel('y'); 
        
        legend_added = True
        #
        for density in QKVs_density_as_key.keys():
            Q,K,V = QKVs_density_as_key[density]
            for laneid in Q.keys():
                #
                if legend_added:
                #color = np.random.uniform(size = (3,))
                    #
                    ax.plot([np.mean(K[laneid])], [np.mean(Q[laneid])],'.', color = color, alpha = alpha, label = label)
                    #
                    legend_added = False
                else:
                    ax.plot([np.mean(K[laneid])], [np.mean(Q[laneid])],'.', color = color, alpha = alpha)
        
        #
        return ax

    @classmethod
    def plot_QK_mean_laneid_as_key(self, Q,K, ax = False,  figsize = (5,5), color = 'b', alpha = .5, label = '2'):
        """
        Q,V,K = TwoDimMicroModel.Get_QVK_from_snapshots(self.snapshots, self.length)
        
            
            Q[laneid] = [q1, q2, q3, q4...]
            K[laneid] = [k1, k2, k3, k4...]
            V[laneid] = [v1,v2,....]
            Q unit is veh/h
            K unit is veh/km
        
        """
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)
            ax.set_xlabel('x');ax.set_ylabel('y'); 
        
        legend_added = True
        #
        for laneid in Q.keys():
            #
            if legend_added:
            #color = np.random.uniform(size = (3,))
                #
                ax.plot([np.mean(K[laneid])], [np.mean(Q[laneid])],'.', color = color, alpha = alpha, label = label)
                #
                legend_added = False
            else:
                ax.plot([np.mean(K[laneid])], [np.mean(Q[laneid])],'.', color = color, alpha = alpha)
        
        #
        return ax
        
        #pass

    @classmethod
    def plot_QK_laneid_as_key(self, Q,K, ax = False,  figsize = (5,5), color = 'b', alpha = .5):
        """
        Q,V,K = TwoDimMicroModel.Get_QVK_from_snapshots(self.snapshots, self.length)
        
            
            Q[laneid] = [q1, q2, q3, q4...]
            K[laneid] = [k1, k2, k3, k4...]
            V[laneid] = [v1,v2,....]
            Q unit is veh/h
            K unit is veh/km
        
        """
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)
            ax.set_xlabel('x');ax.set_ylabel('y'); 
        
        #
        for laneid in Q.keys():
            #
            #color = np.random.uniform(size = (3,))
            ax.plot(K[laneid], Q[laneid],'.', color = color, alpha = alpha)
        
        #
        return ax
        
        #pass

    @classmethod
    def plot_sim_im(self, SNAPSHOTS, ax = False, figsize = (5,3), alpha = .4,):
        """
        
        @input: STATES
        
            STATES[moment] = {vid:np.array([x,vx, y, vy, zlon, zlat])}
        
        ----------------
        - collect the vid and its starting moment t
        - collect the xy
        - collect
        
        """
        
        
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)
            ax.set_xlabel('x');ax.set_ylabel('y'); 
            #ax.grid();
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        
        #vids_startmoments[vid] = t, means that the vid first appear at moment t. 
        vids_startmoments = {}
        vids_existing = set()
        for t in sorted(SNAPSHOTS.keys()):
            #SNAPSHOTS[t].keys are the vids
            vids = set(SNAPSHOTS[t].keys())
            #
            #find the 
            new_vids = vids.difference(vids_existing)
            for vid in new_vids:
                vids_startmoments[vid] = t
                
            vids_existing = vids
        
        #
        for vid in vids_startmoments:
            #
            t = vids_startmoments[vid]
            #find the keys (effective_ts) that vid have data. 
            tmp = np.array(sorted(SNAPSHOTS.keys()))
            effective_ts =tmp[tmp>=t]
            
            #
            xs = [SNAPSHOTS[t][vid]['S'][0] for t  in effective_ts]
            ys = [SNAPSHOTS[t][vid]['S'][2] for t  in effective_ts]
            ax.plot(xs, ys, '.')
            #ax = self.plot_path(path = sim_res, ax = ax, figsize = figsize, alpha = alpha,)
        
        #
        ax.set_xlabel('x ( m )');ax.set_ylabel('y ( m )'); ax.grid()
        
            
        plt.tight_layout()
        
        return ax


    @classmethod
    def plot_sim_heatmap_from_snapshot(self, SNAPTHOTS, ax = False, figsize = (5,3), sigma= 5, alpha = .4,bins_x = 300, bins_y = 40, idx_x = 0, idx_y = 2):
        """
        
        @input: SNAPTHOTS
        
                    #SNAPTHOTS
                    self.snapshots[T]['vehs_at_lanes'] = copy.deepcopy(self.vehs_at_lanes)
                    self.snapshots[T][vid] = {'S':copy.deepcopy(self.vehs_dict[vid]), \
                        #'feasibleincrement':copy.deepcopy(feasible_increment), \
                        'L':copy.deepcopy(L_dicts[vid]), \
                        'F':copy.deepcopy(F_dicts[vid]), \
                        'vehs_info_within_area_left':copy.deepcopy(vehs_info_within_area_left_dict[vid]), \
                        'vehs_info_within_area_right':copy.deepcopy(vehs_info_within_area_right_dict[vid]), \
                        'potentials':copy.deepcopy(potentials_dict[vid]), }
                        
        """
        import matplotlib.cm as cm
        from scipy.ndimage.filters import gaussian_filter

        def myplot(x, y, sigma= 5, bins=(bins_x, bins_y)):
            #
            heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
            heatmap = gaussian_filter(heatmap, sigma=sigma)

            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            return heatmap.T, extent

        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)
            ax.set_xlabel('x');ax.set_ylabel('y'); 
            #ax.grid();
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)
        #keys are lane ids
        t0 = sorted(SNAPTHOTS.keys())[0]
        XS = {laneid:[] for laneid in SNAPTHOTS[t0]['vehs_at_lanes'].keys()}
        YS = {laneid:[] for laneid in SNAPTHOTS[t0]['vehs_at_lanes'].keys()}
        for t in sorted(SNAPTHOTS.keys()):
            #[SNAPTHOTS[t][vid]['S'][0] for vid in SNAPTHOTS[t].keys() ]
            for laneid in SNAPTHOTS[t]['vehs_at_lanes'].keys():
                #
                xs = []
                ys = []
                #
                for vid in SNAPTHOTS[t]['vehs_at_lanes'][laneid]:
                    xs.append(SNAPTHOTS[t][vid]['S'][idx_x])
                    ys.append(SNAPTHOTS[t][vid]['S'][idx_y])
                XS[laneid].extend(xs)
                YS[laneid].extend(ys)
        
        for laneid in XS.keys():
            #print(max(XS) , min(XS), max(YS), min(YS))
            img, extent = myplot(x = XS[laneid], y = YS[laneid], sigma = sigma, bins=(bins_x, bins_y))
            #print(img.shape)
            ax.imshow(img, extent = extent, origin='lower', cmap=cm.jet,  aspect='auto', interpolation='nearest')
        """
        XS = []
        YS = []
        for t in sorted(SNAPTHOTS.keys()):
            #[SNAPTHOTS[t][vid]['S'][0] for vid in SNAPTHOTS[t].keys() ]
            xs = []
            ys = []
            for vid in SNAPTHOTS[t].keys():
                if vid=='vehs_at_lanes':continue
                xs.append(SNAPTHOTS[t][vid]['S'][idx_x])
                ys.append(SNAPTHOTS[t][vid]['S'][idx_y])
            XS.extend(xs)
            YS.extend(ys)
        #
        #print(max(XS) , min(XS), max(YS), min(YS))
        img, extent = myplot(x = XS, y = YS, sigma = sigma, bins=(bins_x, bins_y))
        print(img.shape)
        ax.imshow(img, extent = extent, origin='lower', cmap=cm.jet,  aspect='auto', interpolation='nearest')
        
        
        """
        #
        #heatmap, xedges, yedges = np.histogram2d(XS, YS, bins= (bins_x, bins_y))
        #
        #extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        #ax.imshow(heatmap.T, extent=extent, origin='lower')
        #
        #ax.set_xlabel('x ( m )');ax.set_ylabel('y ( m )'); ax.grid()
        
            
        plt.tight_layout()
        
        return ax





    @classmethod
    def plot_sim_scatter(self, STATES, ax = False, figsize = (5,3), alpha = .4,):
        """
        
        @input: STATES
        
            STATES[moment] = {vid:np.array([x,vx, y, vy, zlon, zlat])}
        
        
        """
        
        
        if isinstance(ax, bool):
            fig,ax = plt.subplots(figsize = figsize)
            ax.set_xlabel('x');ax.set_ylabel('y'); 
            #ax.grid();
            #ax = host_subplot(111)
            #par = ax.twinx()
            #fig,ax = plt.subplots(figsize = figsize, nrows = 1, ncols = 1)

        for t in sorted(STATES.keys()):
            
            
            
            
            
            xs = [STATES[t][vid][0] for vid in STATES[t].keys()]
            ys = [STATES[t][vid][2] for vid in STATES[t].keys()]
            
            ax.plot(xs, ys, '.')
            #ax = self.plot_path(path = sim_res, ax = ax, figsize = figsize, alpha = alpha,)
        
        #
        ax.set_xlabel('x ( m )');ax.set_ylabel('y ( m )'); ax.grid()
        
            
        plt.tight_layout()
        
        return ax


    @classmethod
    def Potential_OV_Peng(self, v_self= 10, v_leader = 10, deltax =20, OV_paras = OV_paras, p = .9):
        """
        the optimal velocity parameters, in Peng, Guanghan, Hongdi He, and Wei-Zhen Lu. 2016. “A New Car-Following Model with the Consideration of Incorporating Timid and Aggressive Driving Behaviors.” Physica A: Statistical Mechanics and Its Applications 442 (January): 197–202. https://doi.org/10.1016/j.physa.2015.09.009.
        
        OV_paras = {'beta':0.65, 'v0':17.65, 'sc':8.2, 'alpha':1.85, 'sigma0_Ngoduy':0.88, \
                'peng_alpha':0.85, 'peng_V1':6.75, 'peng_V2':7.91, 'peng_C1':0.13, 'peng_C2':1.57, 'peng_lc':5, 'lambda_peng':0.4, }
        
        @input: p
        
            a probability indicating aggressiveness. 
            
            greater p means more aggressiveness. 
        
        """
        #
        v_self=v_self/3.6
        v_leader = v_leader/3.6
        #vf = idm_vf/3.6
        
        alpha = OV_paras['peng_alpha']
        V1 = OV_paras['peng_V1']
        V2 = OV_paras['peng_V2']
        C1 = OV_paras['peng_C1']
        C2 = OV_paras['peng_C2']
        lc = OV_paras['peng_lc']
        lambdaa = OV_paras['lambda_peng']
        #
        OV = V1 + V2*np.tanh(C1*(deltax - lc) - C2)
        #
        diff_OV = C1*V2*(1-np.tanh(C1*(deltax - lc) - C2)**2)
        
        return alpha*(OV - v_self) + lambdaa*(v_leader - v_self) + (2*p - 1)*alpha*diff_OV*(v_leader - v_self)
        
        #OV = v0/2.0*(np.tanh(deltax/sc - alpha) + np.tanh(alpha))
        return beta*(OV - v_self)

    

    @classmethod
    def Potential_OV_Ngoduy(self, v_self= 10, v_leader = 10, deltax =20, OV_paras = OV_paras):
        """
        the optimal velocity parameters, in Ngoduy, D., S. Lee, M. Treiber, M. Keyvan-Ekbatani, and H. L. Vu. 2019. “Langevin Method for a Continuous Stochastic Car-Following Model and Its Stability Conditions.” TRANSPORTATION RESEARCH PART C-EMERGING TECHNOLOGIES 105 (August): 599–610. https://doi.org/10.1016/j.trc.2019.06.005.
        
        
        OV_paras= {'beta':0.5, 'v0':25, 'sc':20, 'alpha':2}
        """
        #
        v_self=v_self/3.6
        v_leader = v_leader/3.6
        #vf = idm_vf/3.6
        
        beta = OV_paras['beta']
        alpha = OV_paras['alpha']
        v0 = OV_paras['v0']
        sc = OV_paras['sc']
        
        OV = v0/2.0*(np.tanh(deltax/sc - alpha) + np.tanh(alpha))
        return beta*(OV - v_self)
    
    @classmethod
    def Potential_OV(self, v_self= 10, v_leader = 10, deltax =20, OV_paras = OV_paras):
        """
        the optimal velocity parameters, 
        
        
        """
        #
        idm_vf = idm_paras['idm_vf']
        idm_T = idm_paras['idm_T']
        idm_delta = idm_paras['idm_delta']
        idm_s0 = idm_paras['idm_s0']
        idm_a = idm_paras['idm_a']
        idm_b = idm_paras['idm_b']
        
        #
        v_self=v_self/3.6
        v_leader = v_leader/3.6
        #vf = idm_vf/3.6
        
        #builtins.tmp = v_self,v_leader
        
        try:
            #print()
            #
            s_star = idm_s0+v_self*idm_T+v_self*(v_self - v_leader)/(2.0*np.sqrt(idm_a*idm_b))
            #
            a = 1.0*idm_a*(1-np.power(v_self/idm_vf, idm_delta)-(s_star*s_star)/(deltax*deltax))
            
        except Exception as e:
            
            print('deltax = ',deltax,', v_self=',v_self,', v_leader',v_leader)
            raise ValueError(e)
            
        return a

    
    
    @classmethod
    def Potential_IDM(self, v_self= 10, v_leader = 10, deltax =20, idm_paras = idm_paras, m_buffer = .02, exclude_veh_len = False):
        """
        IDM formation output, the acceleration. 
        
        Note that the delta_v is defined as v_follower-v_leader. 
        
        @type v: float, unit is km/h
        
        @type vf: km/h.
        
        
        @type T: float.
        @param: T, unit is sec
            Average safe time headway.
            
        @type delta: delta:float
        @param: delta
            parameter in the model.
            
            
        @type s0:float
        @param: s0
            parameter 
        
        @OUTPUT: a
            unit is m/s2.
        """
        #
        idm_vf = idm_paras['idm_vf']
        idm_T = idm_paras['idm_T']
        idm_delta = idm_paras['idm_delta']
        idm_s0 = idm_paras['idm_s0']
        idm_a = idm_paras['idm_a']
        idm_b = idm_paras['idm_b']
        veh_len = idm_paras['veh_len']
        #
        v_self=v_self/3.6
        v_leader = v_leader/3.6
        #
        #vf = idm_vf/3.6
        if exclude_veh_len:
            deltax = max(deltax - idm_paras['veh_len'], m_buffer)
        #builtins.tmp = v_self,v_leader
        
        try:
            #print()
            #
            s_star = idm_s0+v_self*idm_T+v_self*(v_self - v_leader)/(2.0*np.sqrt(idm_a*idm_b))
            #
            a = 1.0*idm_a*(1-np.power(v_self/idm_vf, idm_delta)-(s_star*s_star)/(deltax*deltax))
            
        except Exception as e:
            
            print('deltax = ',deltax,', v_self=',v_self,', v_leader',v_leader)
            raise ValueError(e)
            
        return a

        ##########################DEBUG
        v_self=v_self/3.6
        v_leader = v_leader/3.6
        vf = idm_vf/3.6
        
        builtins.tmp = v_self,v_leader
        
        try:
            s_star = idm_s0+v_self*idm_T+v_self*(v_self - v_leader)/(2.0*np.sqrt(idm_a*idm_b))
            a = 1.0*idm_a*(1-np.power(v_self/vf, idm_delta)-(s_star*s_star)/(deltax*deltax))
        except Exception as e:
            
            print('deltax = ',deltax,', v_self=',v_self,', v_leader',v_leader)
            raise ValueError(e)
            
        return a




    @classmethod
    def L(self, state, \
            two_dim_paras = two_dim_paras, \
            idm_paras = idm_paras, \
            stochastic_proecess_name = 'OU',\
            right_mark_force = True,  \
            left_bound_force = True, \
            left_mark_force = True, 
            right_bound_force = True ):
        """
        
        dS = F(S) dt + L(S) dW
        
        """
        sigma_long = two_dim_paras['sigma_long']
        sigma_lat = two_dim_paras['sigma_lat']
        
        #
        x,vx,y, vy,Z_long,Z_lat = state[0],state[1],state[2],state[3],state[4],state[5]
        
        #------------------------------------------
        if stochastic_proecess_name=='OU':
            tmp_long = sigma_long
            tmp_lat = sigma_lat
            #
        elif stochastic_proecess_name=='simple':
            #diff_Z_long = -(sigma_long**1.0)*(Z_long**3)
            #diff_Z_lat = -(sigma_lat**1.0)*(Z_lat)
            #print(Z_long, sigma_long_drift)
            tmp_long =  sigma_long#*min(1.0, np.sqrt(vx**2 + vy**2)/idm_paras['idm_vf'])
            tmp_lat = sigma_lat#*min(1, np.sqrt(vx**2 + vy**2)/idm_paras['idm_vf'])
            
        elif stochastic_proecess_name=='converted':
            #the converted. 
            tmp_long =  sigma_long*(1.0-Z_long**2)
            tmp_lat = sigma_lat*(1-Z_lat**2)
            #
        elif stochastic_proecess_name=='geometric':
            #
            tmp_long =  sigma_long*(Z_long)
            tmp_lat = sigma_lat*(Z_lat)
            #
        elif stochastic_proecess_name=='hyperparabolic':
            #
            tmp_long = sigma_long
            tmp_lat = sigma_lat
            
        elif stochastic_proecess_name=='jacobi':
            #new_state = STATES[-1] - theta*(STATES[-1])*deltat + sigma*np.sqrt((STATES[-1]+.5)*(.5-STATES[-1]))*brownian
            Z_long = max(-.499999, min(Z_long, .4999999))
            tmp_long =   np.sqrt(sigma_long*(Z_long+.5)*(.5-Z_long))
            #print(Z_lat, (.5-Z_lat), (Z_lat+.5))
            Z_lat = max(-.499999, min(Z_lat, .4999999))
            tmp_lat = np.sqrt(sigma_lat*(Z_lat+.5)*(.5-Z_lat))
        elif stochastic_proecess_name=='ROU':
            #
            tmp_long =  sigma_long
            tmp_lat = sigma_lat
        
        #
        #tmp_long =  sigma_long*(1.0-Z_long**2)
        #tmp_lat = sigma_lat*(1-Z_lat**2)
        
        array = np.array([[.0, .0, .0, .0, tmp_long, .0], \
                          [.0, .0, .0, .0, 0, tmp_lat]])
        
        
        return array.T
    
    @classmethod
    def Sample_two_dim_paras(self, ):
        """
        Sample the paramters such that the hetegreoenesou 
        
        """
        
        
        
        pass

    @classmethod
    def PotentialLaneMiddleLine(self, ego_state, ego_lane_lw, lane_middle_line_coor, two_dim_paras = two_dim_paras):
        """
        
        If ego_lateral>lane_middle_line_coor, return a negative value. 
        
        ego_state is defined as:
        
            x,vx,y,vy,noise_lon,noise_lat
        --------------------------------
        @input: lane_middle_line_coor
        
            a float. 
        
        """
        beta = two_dim_paras['beta_lane_middle_line']
        
        #ego_state[2] is y. 
        #if ego_state[2]>lane_middle_line_coor:
        deltay = ego_state[2]-lane_middle_line_coor
        #
        tmp0 = beta*beta*(2.0*deltay-ego_lane_lw)*np.exp(-beta*beta*(deltay-ego_lane_lw/2.0)*(deltay-ego_lane_lw/2.0))
        tmp1 = beta*beta*(2.0*deltay+ego_lane_lw)*np.exp(-beta*beta*(deltay+ego_lane_lw/2.0)*(deltay+ego_lane_lw/2.0))
        #ys2 =  beta*beta*(2.0*deltay-ego_lane_lw)*np.exp(-beta*beta*(deltay-ego_lane_lw/2.0)*(deltay-ego_lane_lw/2.0)) - \
        #        beta*beta*(2.0*deltay+lw)*np.exp(-beta*beta*(deltay+lw/2.0)*(deltay+lw/2.0))
        
        
        return (tmp0 - tmp1)*ego_state[3]

        ################################################################
        ################################################################
        
        beta = two_dim_paras['beta_lane_middle_line']
        #ego_state[2] is y. 
        #if ego_state[2]>lane_middle_line_coor:
        #deltay = ego_state[2]-lane_middle_line_coor
        deltay = y -lane_middle_line_coor
        #
        tmp0 = beta*beta*(2.0*deltay-ego_lane_lw)*np.exp(-beta*beta*(deltay-ego_lane_lw/2.0)*(deltay-ego_lane_lw/2.0))
        tmp1 = beta*beta*(2.0*deltay+ego_lane_lw)*np.exp(-beta*beta*(deltay+ego_lane_lw/2.0)*(deltay+ego_lane_lw/2.0))
        #ys2 =  beta*beta*(2.0*deltay-ego_lane_lw)*np.exp(-beta*beta*(deltay-ego_lane_lw/2.0)*(deltay-ego_lane_lw/2.0)) - \
        #        beta*beta*(2.0*deltay+lw)*np.exp(-beta*beta*(deltay+lw/2.0)*(deltay+lw/2.0))
        return (tmp0 - tmp1)*u


    
    @classmethod
    def PotentialLaneMiddleLine_BKP(self, ego_state, ego_lane_lw, lane_middle_line_coor, two_dim_paras = two_dim_paras):
        """
        
        If ego_lateral>lane_middle_line_coor, return a negative value. 
        
        --------------------------------
        @input: lane_middle_line_coor
        
            a float. 
        
        """
        beta = two_dim_paras['beta_lane_middle_line']
        
        #ego_state[2] is y. 
        
        #if ego_state[2]>lane_middle_line_coor:
        deltay = ego_state[2]-lane_middle_line_coor
        #
        return -ego_lane_lw/2.0*beta*deltay*np.exp(-beta*beta*deltay*deltay)

    
    @classmethod
    def Potential_lw_base(self, ego_state, \
            vehs_info_within_area_left, \
            vehs_info_within_area_right, \
            ego_lane_lw, \
            ego_lane_middle_line_coor, \
            ego_lane_marks_coor, \
            road_bounds, \
            lw_boundary_lanes, \
            leader_state, \
            ellipse_x, ellipse_y_left, ellipse_y_right, \
            two_dim_paras = two_dim_paras, \
            idm_paras = idm_paras, \
            right_mark_force = True,  \
            left_bound_force = True, \
            left_mark_force = True, 
            right_bound_force = True, \
            lanemark_force_type = 'parabolic', \
            lw_base = 3.5):
        """
        Difference between:
        
            - self.Potential()
            - self.Potential_lw_base()
        
        The latter have a specfal base lane width parameter: lw_base
        
        The latter is for the case of lane width, 
        
        
        
        Calculate the potential of the vehicle. 
        
        @input: ego_state
        
            ego_state
        
            a  state of the ego vehicle. 
            
            
            The state of the vehicle is represented as:
        
            - np.array([x, vx, y, vy, zlon, zlat])  x is the longitudinal, y is the lateral, z is the noise. 
        
        @input: lw_boundary_lanes
        
            tuple. 
            
        
            leftmost lane width and rightmost lane width. 
        
        @input: lanes_marks_coors
            
            lanes_marks_coors = (left, right). 
            
            The coordinate of left mark and right mark. 
        
        @input: neighbores_states_dict
        
            neighbores_states_dict[neighbor_id] = state_dict. 
            
            state_dict is a dictionary containing the np.array([x, vx, y, vy, zlon, zlat])
        -------------------------------------------------
        @OUTPUT: potentials
        
            potentials['boundary'], a float
            potentials['lanemark'], a float
            potentials['idm'], a float
            potentials['lanemiddleline'], a float.
            potentials['neighboringlongitudinal'], a float
            potentials['neighboringlateral'], a float.
        
        """
        #x, vx, y, vy, zlon, zlat = 
        #=============================
        potentials = {}
        #
        #------------------potentials['boundary']
        potentials['boundary'] = self.PotentialRoadBoundary(ego_state = ego_state, road_bounds = road_bounds, lw_boundary_lanes = lw_boundary_lanes, two_dim_paras = two_dim_paras, right_force_exist = right_bound_force, left_force_exist = left_bound_force)
        #
        #-----------------potentials['lanemark']
        if lanemark_force_type=='parabolic':
            potentials['lanemark'] = self.PotentialLaneMark_parabolic(ego_state = ego_state, ego_lane_marks_coor = ego_lane_marks_coor, two_dim_paras = two_dim_paras, right_force_exist = right_mark_force, left_force_exist = left_mark_force)
        elif lanemark_force_type=='exponential':
            potentials['lanemark'] = self.PotentialLaneMark(ego_state = ego_state, ego_lane_marks_coor = ego_lane_marks_coor, two_dim_paras = two_dim_paras, right_force_exist = right_mark_force, left_force_exist = left_mark_force)
        #
        #-----------------potentials['idm']
        potentials['idm'] = self.Potential_IDM(v_self= ego_state[1]*3.6, v_leader = leader_state[1]*3.6, deltax =abs(leader_state[0] - ego_state[0]), idm_paras = idm_paras)
        #
        #-------------------potentials['lanemiddleline']
        potentials['lanemiddleline'] = self.PotentialLaneMiddleLine(ego_state  = ego_state, \
            ego_lane_lw = ego_lane_lw, \
            lane_middle_line_coor = ego_lane_middle_line_coor, \
            two_dim_paras = two_dim_paras)
        #
        #-----------------potentials['neighboringlongitudinal']
        longitudinalforce,lateralforce = self.PotentialNeighboringLongitudinalLateral_lw_base(ego_state = ego_state, \
            ego_lane_lw = ego_lane_lw, \
            vehs_info_within_area_left = vehs_info_within_area_left, \
            vehs_info_within_area_right = vehs_info_within_area_right, \
            ellipse_x = ellipse_x, \
            ellipse_y_left  = ellipse_y_left, \
            ellipse_y_right = ellipse_y_right, \
            lw_base = lw_base)
        #
        #
        potentials['neighboringlongitudinal'] = longitudinalforce
        potentials['neighboringlateral'] = lateralforce
        
        
        return potentials


    
    @classmethod
    def Potential(self, ego_state, \
            vehs_info_within_area_left, \
            vehs_info_within_area_right, \
            ego_lane_lw, \
            ego_lane_middle_line_coor, \
            ego_lane_marks_coor, \
            road_bounds, \
            lw_boundary_lanes, \
            leader_state, \
            ellipse_x, ellipse_y_left, ellipse_y_right, \
            two_dim_paras = two_dim_paras, \
            idm_paras = idm_paras, \
            right_mark_force = True,  \
            left_bound_force = True, \
            left_mark_force = True, 
            right_bound_force = True, \
            lanemark_force_type = 'exponential'):
        """
        Calculate the potential of the vehicle. 
        
        @input: ego_state
        
            ego_state
        
            a  state of the ego vehicle. 
            
            
            The state of the vehicle is represented as:
        
            - np.array([x, vx, y, vy, zlon, zlat])  x is the longitudinal, y is the lateral, z is the noise. 
        
        @input: lw_boundary_lanes
        
            tuple. 
            
        
            leftmost lane width and rightmost lane width. 
        
        @input: lanes_marks_coors
            
            lanes_marks_coors = (left, right). 
            
            The coordinate of left mark and right mark. 
        
        @input: neighbores_states_dict
        
            neighbores_states_dict[neighbor_id] = state_dict. 
            
            state_dict is a dictionary containing the np.array([x, vx, y, vy, zlon, zlat])
        -------------------------------------------------
        @OUTPUT: potentials
        
            potentials['boundary'], a float
            potentials['lanemark'], a float
            potentials['idm'], a float
            potentials['lanemiddleline'], a float.
            potentials['neighboringlongitudinal'], a float
            potentials['neighboringlateral'], a float.
        
        """
        #x, vx, y, vy, zlon, zlat = 
        #=============================
        potentials = {}
        #
        #------------------potentials['boundary']
        potentials['boundary'] = self.PotentialRoadBoundary(ego_state = ego_state, road_bounds = road_bounds, lw_boundary_lanes = lw_boundary_lanes, two_dim_paras = two_dim_paras, right_force_exist = right_bound_force, left_force_exist = left_bound_force)
        #
        #-----------------potentials['lanemark']
        if lanemark_force_type=='parabolic':
            potentials['lanemark'] = self.PotentialLaneMark_parabolic(ego_state = ego_state, ego_lane_marks_coor = ego_lane_marks_coor, two_dim_paras = two_dim_paras, right_force_exist = right_mark_force, left_force_exist = left_mark_force)
        elif lanemark_force_type=='exponential':
            potentials['lanemark'] = self.PotentialLaneMark(ego_state = ego_state, ego_lane_marks_coor = ego_lane_marks_coor, two_dim_paras = two_dim_paras, right_force_exist = right_mark_force, left_force_exist = left_mark_force)
        #
        #-----------------potentials['idm']
        potentials['idm'] = self.Potential_IDM(v_self= ego_state[1]*3.6, v_leader = leader_state[1]*3.6, deltax =abs(leader_state[0] - ego_state[0]), idm_paras = idm_paras)
        #
        #-------------------potentials['lanemiddleline']
        potentials['lanemiddleline'] = self.PotentialLaneMiddleLine(ego_state  = ego_state, \
            ego_lane_lw = ego_lane_lw, \
            lane_middle_line_coor = ego_lane_middle_line_coor, \
            two_dim_paras = two_dim_paras)
        #
        #-----------------potentials['neighboringlongitudinal']
        longitudinalforce,lateralforce = self.PotentialNeighboringLongitudinalLateral(ego_state = ego_state, \
            ego_lane_lw = ego_lane_lw, \
            vehs_info_within_area_left = vehs_info_within_area_left, \
            vehs_info_within_area_right = vehs_info_within_area_right, \
            ellipse_x = ellipse_x, \
            ellipse_y_left  = ellipse_y_left, \
            ellipse_y_right = ellipse_y_right, )
        #
        potentials['neighboringlongitudinal'] = longitudinalforce
        potentials['neighboringlateral'] = lateralforce
        
        
        return potentials





def calculate_ngsim_speed_acceleration(ngsim):
    '''
        Calculate the velocity and acceleration 
        for the NGSIM dataset 
    '''

    vehicles = []

    for id in data.id.unique():

        # # Get the vehicle's trajectory
        veh = data[data['id']==id]
        veh.sort_values(['id','frame'],axis=0,ascending=True,inplace=True)

        veh.x.shape, np.diff(veh.x).shape

        # # Calculate the time step
        dt = 0.1 * veh.frame.diff()

        # # Calculate the velocity and acceleration
        xVelocity = veh.x.diff() / dt
        yVelocity = veh.y.diff() / dt
        xAcceleration = xVelocity.diff() / dt
        yAcceleration = yVelocity.diff() / dt

        veh['xVelocity'] = xVelocity
        veh['yVelocity'] = yVelocity
        veh['xAcceleration'] = xAcceleration
        veh['yAcceleration'] = yAcceleration

        veh = veh.dropna(axis=0, subset=['yAcceleration'])
        vehicles.append(veh)

    # # splicing each trajectory
    vehicles = pd.concat(vehicles)
    
    return vehicles


def rename_dataframe(dataset):
    '''
        Rename the dataset
    '''
    n_col = dataset.shape[1]
    
    if (n_col == 25) and ('Section_ID' in dataset.columns.tolist()):
        dataset.rename(columns={"v_length":"length", "v_length":"width", 'v_Vel':'Velocity', 'Local_X':'y',
                                'lane':'laneId', 'Lane_ID':'laneId', 'Vehicle_ID':'id', 'Frame_ID':'frame',
                                'Local_Y':'x'}, inplace=True)
        
        # calculate velocity and acceleration
        dataset = calculate_ngsim_speed_acceleration(dataset)
        
    elif n_col == 26:
        dataset.rename(columns={"width":"length", "height":"width"}, inplace=True)
    
    elif n_col == 16:
        dataset.rename(columns={"frameId":"frame", "trackId":"id", 'laneId':'laneId', 'xVelocity':'yVelocity',
                               'yVelocity':'xVelocity', 'localY':'x', 'localX':'y', 'yAcceleration':'xAcceleration',
                               'xAcceleration':'yAcceleration'}, inplace=True) 

    return dataset


def filter_lane_keep_data(full_data):
    '''
        Filter the lane-keep data 
        from the full data
    '''
    vids = full_data.id.unique()

    ## filter lane-keep
    vehicles = []
    for id in vids:
        veh = full_data[full_data['id']==id]
        if veh.laneId.unique().shape[0] == 1:
            vehicles.append(veh)

    ## concat these sub-dataframe
    vehicles = pd.concat(vehicles)
    
    return vehicles


def data_process(data):
    '''
        Process dataframe: 1) add class;
        2) rename dataframe.
    '''
    ## Add vehicle's class
    tqdm.pandas()
    data['class'] = data['id'].progress_apply(lambda x: attribute[attribute.id==x]['class'].tolist()[0])

    # ## Rename these columns
    data = rename_dataframe(data)
    
    return data


def calculate_jerk(data, dt):
    '''
        Calculate vehicle's jerk.
    '''
    ## Calculate jerk
    vids = data.id.unique()
    VEHICLE = []
    for id in tqdm(vids):
        veh = data[data.id==id]

        ## the forward direction
        if veh.laneId.isin([6,7,8]).unique()[0]:
            veh['xJerk'] = veh.xAcceleration.diff() / dt
            veh['yJerk'] = veh.yAcceleration.diff() / dt
             
        ## the backward direction
        else:
            veh['xJerk'] = -veh.xAcceleration.diff() / dt
            veh['yJerk'] = -veh.yAcceleration.diff() / dt

        # if veh.laneId.isin([6,7,8]).unique()[0]:
            # veh['xJerk'] = veh.x.diff().diff().diff() / (dt**3)
            # veh['yJerk'] = veh.y.diff().diff().diff() / (dt**3)
        # the backward direction
        # else:
            # veh['xJerk'] = -veh.x.diff().diff().diff() / (dt**3)
            # veh['yJerk'] = -veh.y.diff().diff().diff() / (dt**3)


        ## collect vehicles
        VEHICLE.append(veh.iloc[3:])

    ## Concat vehicles
    VEHICLE = pd.concat(VEHICLE)
    
    return VEHICLE


def plot_results(data,xBins, yBins):
    '''
        Plot jerk results
    '''
    # data = data[(data.yJerk<=1) & (data.yJerk>=-1) & (data.xJerk<=1) & (data.xJerk>=-1)]
    fig, axes = plt.subplots(1, 2, figsize=(12,4.5))
    a = axes[0].hist(data.xJerk, xBins, alpha=0.75, label='Longitudinal Jerk')
    b = axes[1].hist(data.yJerk, yBins, alpha=0.75, label='Lateral Jerk')
    axes[0].set_xlabel('Jerk ($m/s^{3}$)', fontsize = 14)
    axes[0].set_ylabel('Frequency', fontsize = 14)
    axes[0].legend(fontsize=14)
    axes[0].grid()
    # axes[0].set_xlim(-1,1)
    axes[1].set_xlabel('Jerk ($m/s^{3}$)', fontsize = 14)
    axes[1].legend(fontsize=14)
    axes[1].grid()
    # axes[1].set_xlim(-1,1)
    return fig


def Calculate_CACC_Jerk_deltaT(CACC_data, J):
    '''
        Calculate the 'deltaT' for CACC dataset.
    '''
    ##--------------------------calculate CACC's Jerks
    for i in range(1,6):
        CACC_data['Jerk'+str(i)] = CACC_data['Speed'+str(i)].diff().diff() / 0.01

    ##--------------------------Filter CACC's Jerks
    deltaT_, deltaT = {}, []
    for i in tqdm(range(1,6)):
        ## filter time
        time = CACC_data[CACC_data['Jerk'+str(i)]>J].Time.tolist()

        ## calculate time difference
        if len(time)>=2:
            deltaT_[i] = np.diff(time)
            deltaT.append(np.diff(time))

    ## format results
    deltaT = np.concatenate(deltaT)

    return deltaT, deltaT_


def Calculate_highD_Jerk_deltaT(highd_data, J):
    '''
        Calculate the 'deltaT' for highD dataset.
    '''
    ##---------------------------------------calculate the combined Acceleration
    highd_data['Acceleration'] = np.sqrt(highd_data.xAcceleration**2 + highd_data.xAcceleration**2)

    ## filter lane keep data
    laneKeepJerk = filter_lane_keep_data(highd_data)

    ## filter lane keep data
    deltaT, deltaTX, deltaTY = [], [], []
    vids = laneKeepJerk.id.unique()
    for id in tqdm(vids):
        ## get vehicle's track
        veh = highd_data[highd_data.id==id]

        ## calculate vehicle's Jerk
        veh['Jerk'] = veh.Acceleration.diff() / 0.04
        veh['xJerk'] = veh.xAcceleration.diff() / 0.04
        veh['yJerk'] = veh.yAcceleration.diff() / 0.04
        
        ## filter time
        time = veh[veh.Jerk>J].frame.tolist()
        xTime = veh[veh.xJerk>J].frame.tolist()
        yTime = veh[veh.yJerk>J].frame.tolist()

        if len(time)>=2:
            deltaT.append(np.diff(time))
        if len(xTime)>=2:
            deltaTX.append(np.diff(xTime))
        if len(yTime)>=2:
            deltaTY.append(np.diff(yTime))
        
        
    ## format results
    deltaT = np.concatenate(deltaT) * 0.04
    deltaTX = np.concatenate(deltaTX) * 0.04
    deltaTY = np.concatenate(deltaTY) * 0.04

    return deltaT, deltaTX, deltaTY


class Calculate_CACC_Jerk_deltaT00000000000:
    '''
    Calculate the jerk and deltaT 
    distributions for CACC.
    '''
    def __init__(self, CACC_data, P):
        self.CACC_data = CACC_data
        self.P = P
    
    

    def calculate_jerk(self):
        '''
            Calculate Jerks.
        '''
        CACC_data = self.CACC_data
        Jerk, Jerk_ = [], {}
        for i in range(1,6):
            jerk = CACC_data['Speed'+str(i)].diff().diff() / 0.01
            CACC_data['Jerk'+str(i)] = jerk
            Jerk.append(jerk.iloc[2:].values)
            Jerk_[i] = jerk.iloc[2:].values
        
        ## reformat result
        Jerk = np.concatenate(Jerk)
        self.CACC_data = CACC_data.iloc[2:]
        self.Jerk = Jerk

        return Jerk, Jerk_
    

    def calculate_deltaT(self):
        ## calculate the percentile
        self.J = np.percentile(self.Jerk, self.P)

        ## calculate the 'deltaT'
        deltaT = []
        for i in tqdm(range(1,6)):
            ## filter time
            time = self.CACC_data[self.CACC_data['Jerk'+str(i)]>self.J].Time.tolist()

            ## calculate time difference
            if len(time)>=2:
                deltaT.append(np.diff(time))

        ## format results
        deltaT = np.concatenate(deltaT)
        return self.J, deltaT
    
    
    def Run(self):
        ## calculate the Jerk distribution
        Jerk, Jerk_ = self.calculate_jerk()

        ## calculate the deltaT
        J, deltaT = self.calculate_deltaT()
        return Jerk, Jerk_, J, deltaT

from tqdm import tqdm
class Calculate_highD_Jerk_deltaT0():
    '''
        Calculate the jerk and deltaT 
        distributions for highD.
    '''
    def __init__(self, dajiang_data, P):
        highd_data['Acceleration'] = np.sqrt(highd_data.xAcceleration**2 + highd_data.xAcceleration**2)
        self.highd_data = highd_data
        self.laneKeep = filter_lane_keep_data(highd_data)
        self.P = P
    
    def calculate_jerk(self):
        '''
            Calculate Jerks.
        '''
        vids = self.laneKeep.id.unique()
        VEHICLE, Jerk, xJerk, yJerk = [], [], [], []
        for id in tqdm(vids):
            veh = self.highd_data[self.highd_data.id==id]
            ## calculate vehicle's Jerk
            veh['Jerk'] = veh.Acceleration.diff() / 0.04
            veh['xJerk'] = veh.xAcceleration.diff() / 0.04
            veh['yJerk'] = veh.yAcceleration.diff() / 0.04
            VEHICLE.append(veh)

            ## append vehicles
            Jerk.append(veh['Jerk'].iloc[1:].values)
            xJerk.append(veh['xJerk'].iloc[1:].values)
            yJerk.append(veh['yJerk'].iloc[1:].values)
            
        ## format results
        Jerk = np.concatenate(Jerk)
        xJerk = np.concatenate(xJerk)
        yJerk = np.concatenate(yJerk)

        ## become global varibles
        self.Jerk= Jerk
        self.xJerk = xJerk
        self.yJerk = yJerk
        self.VEHICLE = pd.concat(VEHICLE)
        
        return xJerk, yJerk

    def calculate_deltaT(self):
        '''
            Calculate the deltaT.
        '''
        ## calculate the percentile
        self.Jx = np.percentile(self.xJerk, self.P)
        self.Jy = np.percentile(self.yJerk, self.P)

        ## calculate the 'deltaT'
        deltaTX, deltaTY = [], []
        vids = self.VEHICLE.id.unique()
        for id in tqdm(vids):
            ## get vehicle's track
            veh = self.VEHICLE[self.VEHICLE.id==id]
            ## filter time
            xTime = veh[veh.xJerk>self.Jx].frame.tolist()
            yTime = veh[veh.yJerk>self.Jy].frame.tolist()

            if len(xTime)>=2:
                deltaTX.append(np.diff(xTime))
            if len(yTime)>=2:
                deltaTY.append(np.diff(yTime))

        ## format results
        deltaTX = np.concatenate(deltaTX) * 0.04
        deltaTY = np.concatenate(deltaTY) * 0.04

        return self.Jx, self.Jy, deltaTX, deltaTY

    def Run(self):
        ## calculate the Jerk distribution
        xJerk, yJerk = self.calculate_jerk()

        ## calculate the deltaT
        Jx, Jy, deltaTX, deltaTY = self.calculate_deltaT()

        return xJerk, yJerk, Jx, Jy, deltaTX, deltaTY


class Calculate_More_Datasets_Jerk_deltaT:
    '''
        Calculate the jerk and deltaT 
        distributions for datasets.
    '''
    def __init__(self, data, P, dt):
        ## the Dajiang 
        data = data[data.laneId.isin([1,2,3,5,6,7])]
        data['Acceleration'] = np.sqrt(data.xAcceleration**2 + data.xAcceleration**2)
        
        self.data = data
        self.laneKeep = filter_lane_keep_data(data)
        self.P = P
        self.dt = dt
    
    def calculate_jerk(self):
        '''
            Calculate Jerks.
        '''
        vids = self.laneKeep.id.unique()
        VEHICLE, Jerk, xJerk, yJerk = [], [], [], []
        for id in tqdm(vids):
            veh = self.data[self.data.id==id]
            ## calculate vehicle's Jerk
            veh['Jerk'] = veh.Acceleration.diff() / (1/30)
            veh['xJerk'] = veh.xAcceleration.diff() / (1/30)
            veh['yJerk'] = veh.yAcceleration.diff() / (1/30)
            VEHICLE.append(veh)

            ## append vehicles
            Jerk.append(veh['Jerk'].iloc[1:].values)
            xJerk.append(veh['xJerk'].iloc[1:].values)
            yJerk.append(veh['yJerk'].iloc[1:].values)
            
        ## format results
        Jerk = np.concatenate(Jerk)
        xJerk = np.concatenate(xJerk)
        yJerk = np.concatenate(yJerk)

        ## become global varibles
        self.Jerk= Jerk
        self.xJerk = xJerk
        self.yJerk = yJerk
        self.VEHICLE = pd.concat(VEHICLE)
        
        return xJerk, yJerk

    def calculate_deltaT(self):
        '''
            Calculate the deltaT.
        '''
        ## calculate the percentile
        self.Jx = np.percentile(self.xJerk, self.P)
        self.Jy = np.percentile(self.yJerk, self.P)

        ## calculate the 'deltaT'
        deltaTX, deltaTY = [], []
        vids = self.VEHICLE.id.unique()
        for id in tqdm(vids):
            ## get vehicle's track
            veh = self.VEHICLE[self.VEHICLE.id==id]
            ## filter time
            xTime = veh[veh.xJerk>self.Jx].frame.tolist()
            yTime = veh[veh.yJerk>self.Jy].frame.tolist()

            if len(xTime)>=2:
                deltaTX.append(np.diff(xTime))
            if len(yTime)>=2:
                deltaTY.append(np.diff(yTime))

        ## format results
        deltaTX = np.concatenate(deltaTX) * (1/30)
        deltaTY = np.concatenate(deltaTY) * (1/30)

        return self.Jx, self.Jy, deltaTX, deltaTY

    def Run(self):
        ## calculate the Jerk distribution
        xJerk, yJerk = self.calculate_jerk()

        ## calculate the deltaT
        Jx, Jy, deltaTX, deltaTY = self.calculate_deltaT()

        return xJerk, yJerk, Jx, Jy, deltaTX, deltaTY


def FitExponentDistribution(Tx):
    '''
        Fit the Exponent distribution.
    '''
    P = expon.fit(Tx)
    loc, scale = P
    T = np.linspace(0,ceil(Tx.max()),100)
    rP = expon.pdf(T,*P)
    # plt.plot(T,rP)
    return loc, scale, T, rP






















