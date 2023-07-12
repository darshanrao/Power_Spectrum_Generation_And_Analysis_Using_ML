import numpy as np
from py21cmmc import analyse
from py21cmmc import mcmc
import py21cmmc as p21mc
import csv
import random
import argparse
from py21cmfast import cache_tools


parser = argparse.ArgumentParser()
parser.add_argument("no_of_sim",help="Num of generations")
parser.add_argument("f_no",help="File number")
args=parser.parse_args()
n=int(args.no_of_sim)
x=int(args.f_no)

#HII_EFF_FACTOR Range:[5,200]
#ION_Tvir_MIN Range:[4,6]
#R_BUBBLE_MAX Range:[5,20]

i=0
while(i<n):
    try:
        h2_eff= random.uniform(5,200) 
        vir_min= random.uniform(4,6)  
        r_mfp= random.uniform(5,20) 

        #Creating Cores
        core = p21mc.CoreCoevalModule( 
        redshift = 7,         
        user_params = dict(       
            HII_DIM = 50,        
            BOX_LEN = 125.0       
        ),
        flag_options={'USE_MASS_DEPENDENT_ZETA': False},
        astro_params={'HII_EFF_FACTOR':h2_eff,'ION_Tvir_MIN':vir_min,'R_BUBBLE_MAX':r_mfp},
        regenerate=False         
        ) 
        
        filepath="./DATA/data_"+str(i+(n*(x-1)))

        datafiles = [filepath+"/simple_mcmc_data_%s.npz"%z for z in core.redshift]
        info_list=[]

        info_list.append([h2_eff,vir_min,r_mfp])



        #Likelihood Function
        likelihood = p21mc.Likelihood1DPowerCoeval(  
            datafile = datafiles,                   
            noisefile= None,                        
            min_k=0.1,                             
            max_k=1.0,                              
            simulate = True,)                    

        model_name = "SimpleTest"

        chain = mcmc.run_mcmc(
            core, likelihood,        # Use lists if multiple cores/likelihoods required. These will be eval'd in order.
            datadir=filepath,          # Directory for all outputs
            model_name=model_name,   # Filename of main chain output
            params=dict(             # Parameter dict as described above.
                HII_EFF_FACTOR = [h2_eff, h2_eff-0.001, h2_eff+0.001, 0.00001],
                ION_Tvir_MIN = [vir_min, vir_min-0.001, vir_min+0.001, 0.00001],
                R_BUBBLE_MAX = [r_mfp, r_mfp-0.001, r_mfp+0.001, 0.00001]
            ),
            walkersRatio=2,         # The number of walkers will be walkersRatio*nparams
            burninIterations=0,      # Number of iterations to save as burnin. Recommended to leave as zero.
            sampleIterations=1,    # Number of iterations to sample, per walker.
            threadCount=2,           # Number of processes to use in MCMC (best as a factor of walkersRatio)
            continue_sampling=False  # Whether to contine sampling from previous run *up to* sampleIterations.
        )


        #Saving the parameters in CSV format
        fields = ['HII_EFF_FACTOR', 'ION_Tvir_MIN','R_BUBBLE_MAX'] 
        with open(filepath+'/data.csv', 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(fields)
            csv_writer.writerows(info_list)
        print(i)
        print("Done")
        i=i+1
        cachex=cache_tools.clear_cache()
    except:
        pass
    