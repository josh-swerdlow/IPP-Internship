# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 06:12:10 2018

@author: petr
"""

# -*- coding: utf-8 -*-

import logging
import numpy as np
import mpfit
from matplotlib import pyplot
from matplotlib import gridspec
import os
import subprocess
from scipy.io import netcdf
import scipy.interpolate

def readstrahlnetcdf(result):
    file = netcdf.netcdf_file(result, 'r')
    signal = file.variables
    #testsignal = file.variables['impurity_density'].copy()
    dimension = file.dimensions
    attribute = file._attributes
    
    '''
    print('Dimension names and values')
    print('')
    for keys, values in dimension.items():
        print(keys,'...........................',values)
    print('')
    print('')
    
    print('Attribute names and values')
    print('')
    for keys, values in attribute.items():
        print(keys,'...........................',values)
    print('')
    print('')
    
    print('Variable names and shapes')
    print('')
    '''
    data = {}
    empty = ()
    for keys, values in signal.items():
        #print(keys,'...........................',values.shape)
        if values.shape is empty:
            data[keys] = values.data
        else:
            data[keys] = signal[keys][:].copy()
        #data[keys] = {keys : signal[keys][:].copy()}
    #print('')
    #print('')
 
    file.close()
    
    return dimension, data, attribute

class chisquaredfitter:

    def loglinear(self
                  ,coeff2
                  ,x2
                  ,x_offset):
        """
        Evaluate a linear function to the Ln(Nz)
        This is to purely fit the confinement time in STRAHL with no background

        Fitting is done as a function of x
        where x is the time window from max value + offset to users desired end
        """
        
        y = np.zeros(len(x2))
        index = (np.where(x2==x_offset))[0]
        for iii in range(index[0],len(x2)):
            y[iii] = coeff2[0] - coeff2[1]*x2[iii]

        return y

    def loglinearresidual(self, coeff2):
        """
        Return the residual for the loglinearstrahl funciton.
        This method can be bassed to mir_mpfit.
        """

        yfit2 = self.loglinear(coeff2, self.x2, self.x_offset)
        residual2 = (self.data2 - yfit2) / self.sigma2
          
        return {'residual':residual2}

    def loglinearfit(self):
        
        parameter_guess2 = np.array([10.0
                                    ,0.05])

        mp2 = mpfit.mpfit(self.loglinearresidual
                         ,parameter_guess2
                         ,quiet=True)
        
        
        logging.info(mp2.errmsg)
        logging.info('mpfit status:')
        logging.info(mp2.statusString())
        
        if mp2.perror is None: mp2.perror = parameter_guess2*0
            
        yfit2 = self.loglinear(mp2.params
                               ,self.x2
                               ,self.x_offset)
        
        #print(mp2.perror)
        #print(mp2.fnorm)
        print('The fit status is ', mp2.status)
        print('The number of iterations completed: ', mp2.niter)
        print('The number of calls to loglinear: ', mp2.nfev)
        
        return {'type':'loglinear'
                ,'description':'Linear: y[iii] = coeff2[0] + coeff2[1]*x2[iii]'
                ,'summedchisquared':mp2.fnorm
                ,'fit':yfit2
                ,'params':mp2.params
                ,'perror':mp2.perror
                ,'coeff':parameter_guess2
                ,'minparamstepsize':parameter_guess2*0}

    def executestrahl(self
                      ,maininputfilename=None
                      ,directory=None
                      ,difval=None
                      ,numdloc=None
                      ,choiceedgerhod=None
                      ,choiceaxisrhod=None
                      ,convval=None
                      ,numvloc=None
                      ,choiceedgerhov=None
                      ,choiceaxisrhov=None):
        
        file = maininputfilename
        directory = '/draco/u/petr/strahlnew_copy/result/'+file#+'_contour2' #directory
        netcdffilename = 'd' + str(difval[0]) + '_v' + str(convval[1])   
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        command = []
        difstr = str(np.asarray(difval,dtype=np.float32))[1:-1]
        convstr = str(np.asarray(convval,dtype=np.float32))[1:-1]
        command.append('./strahl')
        #command.append('cd result/\n')
        command.append('cp Festrahl_result.dat '+ netcdffilename + '\n')
        command.append('mv '+ netcdffilename + ' ' + directory + '\n')
        
        rhodval = np.arange(choiceaxisrhod,(choiceedgerhod-choiceaxisrhod)+((choiceedgerhod-choiceaxisrhod)/(numdloc-1)),(choiceedgerhod-choiceaxisrhod)/(numdloc-1))      
        rhodvalstr = (np.array_str(rhodval,max_line_width=1e10))[1:-1]
        rhovval = np.arange(choiceaxisrhov,(choiceedgerhov-choiceaxisrhov)+((choiceedgerhov-choiceaxisrhov)/(numvloc-1)),(choiceedgerhov-choiceaxisrhov)/(numvloc-1))
        rhovvalstr = (np.array_str(rhovval,max_line_width=1e10))[1:-1]

        '''
        print('str(numdloc)', str(numdloc))
        print('rhodvalstr ',rhodvalstr)
        print('difstr ', difstr)
        print('str(numvloc)', str(numvloc))
        print('rhovvalstr ',rhovvalstr)
        print('convstr ', convstr)
        '''
        
        output = subprocess.check_output([command[0]],input='\n'.join([file,'/',str(numdloc),rhodvalstr,difstr,str(numvloc),rhovvalstr,convstr,'e']),universal_newlines=True,stderr=subprocess.STDOUT)
        #subprocess.call(command[0],shell=True,cwd='/draco/u/petr/strahlnew_copy/result/')
        print(output, end='')
        print('STRAHL main input filename: ', file)
        print('D values: ',difstr)
        print('V values: ',convstr)
    
        #changedir = subprocess.check_output([command[1]],universal_newlines=True,cwd='/draco/u/petr/strahlnew_copy/')
        #print(changedir, end='')
        #copyfile = subprocess.check_output([command[2]],universal_newlines=True,cwd='/draco/u/petr/strahlnew_copy/result/')
        #print(copyfile, end='')
        #movefile = subprocess.check_output([command[3]],universal_newlines=True)
        #print(movefile, end='')
        #time.sleep(5)
        subprocess.call(command[1],shell=True,cwd='/draco/u/petr/strahlnew_copy/result/')
        subprocess.call(command[2],shell=True,cwd='/draco/u/petr/strahlnew_copy/result/')
        #subprocess.Popen(command[1],cwd='/draco/u/petr/strahlnew_copy/result/')
        #subprocess.Popen(command[2],cwd='/draco/u/petr/strahlnew_copy/result/')
        totalname = directory+'/'+netcdffilename
        
        return {'totalname':totalname, 'rhodval':rhodval, 'rhovval':rhovval}

    def loopstrahl(self
                  ,maininputfilename=None
                  ,directory=None
                  ,numofdifval=None
                  ,lowestdifval=None
                  ,highestdifval=None
                  ,numofconvval=None
                  ,lowestconvval=None
                  ,highestconvval=None):
     
        file = maininputfilename
        difvalarray = np.linspace(lowestdifval,highestdifval,num=numofdifval)
        convvalarray = np.linspace(lowestconvval,highestconvval,num=numofconvval)

        for ii in range(len(difvalarray)):
            for iii in range(len(convvalarray)):
                totalpathname = self.executestrahl(maininputfilename=file,difval=[difvalarray[ii],difvalarray[ii]],convval=[convvalarray[iii],convvalarray[iii]])

    def strahlmodelsignal(self
                          ,coeff
                          ,x=None
                          ,numdloc=None
                          ,choiceedgerhod=None
                          ,choiceaxisrhod=None
                          ,numvloc=None
                          ,choiceedgerhov=None
                          ,choiceaxisrhov=None):
        

        difval = []
        for kk in range(numdloc):
            difval.append(coeff[kk])

        convval = []
        for jj in range(numvloc):
            convval.append(coeff[numdloc+jj])

        #print('difval ', difval)
        #print('convval ', convval)
        
        strahlexecution = self.executestrahl(maininputfilename='op12a_171123036_FeLBO'
                                             ,difval=difval
                                             ,numdloc=numdloc
                                             ,choiceedgerhod=choiceedgerhod
                                             ,choiceaxisrhod=choiceaxisrhod
                                             ,convval=convval
                                             ,numvloc=numvloc
                                             ,choiceedgerhov=choiceedgerhov
                                             ,choiceaxisrhov=choiceaxisrhov)

        dimension, data, attribute = readstrahlnetcdf(strahlexecution['totalname'])

        #This is for matching to impurity density
        #signal = data['impurity_density'][:,int(self.ionstage)]

        
        '''
        This is for matching to impurity line radiation of Iron (where 24+ is listed in the 6th spot in Fe.atomdat)
        cd   charge of ion      wavelength(A)     half width of window(A)       file extension
        12                203.700                    20.                   'ben'
        18                108.400                    20.                   'ben'
        19                121.800                    20.                   'ben'
        21                117.200                    20.                   'ben'
        22                132.700                    20.                   'ben'
        24                1.85                       0.07                  'llu'
        '''
        numofradlines = len(data['diag_lines_radiation'][0,:,0])
        signalFe12 = data['diag_lines_radiation'][:,0]
        signalFe18 = data['diag_lines_radiation'][:,1]
        signalFe19 = data['diag_lines_radiation'][:,2]
        signalFe21 = data['diag_lines_radiation'][:,3]
        signalFe22 = data['diag_lines_radiation'][:,4]
        signalFe24 = data['diag_lines_radiation'][:,5]

        '''
        if (np.isnan(signalFe12)).any() == True:
            signalFe12[np.isnan(signalFe12)] = -1000

        if (np.isnan(signalFe18)).any() == True:
            signalFe18[np.isnan(signalFe18)] = -1000

        if (np.isnan(signalFe19)).any() == True:
            signalFe19[np.isnan(signalFe19)] = -1000

        if (np.isnan(signalFe21)).any() == True:
            signalFe21[np.isnan(signalFe21)] = -1000

        if (np.isnan(signalFe22)).any() == True:
            signalFe22[np.isnan(signalFe22)] = -1000

        if (np.isnan(signalFe24)).any() == True:
            signalFe24[np.isnan(signalFe24)] = -1000
        '''

        #print(*signalFe24)
        #print(*signalFe22)
        #print('signalFe12')
        #print('data[impurity_density][400,24,:] ',*data['impurity_density'][400,24,:])
        #print(*signalFe12)

        signaltime = data['time']
        signalrho = data['rho_poloidal_grid']

        signalspaceinterp = np.zeros((len(signaltime),len(self.xrho)))
        signalspaceinterpnorm = np.zeros((len(signaltime),len(self.xrho)))
        
        background = np.zeros((len(x),len(self.xrho)))
        signalspacetimeinterp = np.zeros((len(x),len(self.xrho)))
        signalwithbackground = np.zeros((len(x),len(self.xrho)))

     
        for iii in range(len(signaltime)):
            signalspaceinterp[iii] = np.interp(self.xrho,signalrho,signalFe24[iii])
            #print('signalspaceinterp[iii] ',signalspaceinterp[iii],' signalFe24[iii] is ',signalFe24[iii])

        for iiii in range(len(self.xrho)):
            #if signalspaceinterp.max() == 0.0:
                #signalspaceinterp[0,0] = 1.0
            signalspaceinterpnorm[:,iiii] = signalspaceinterp[:,iiii] / signalspaceinterp.max()
            #if (np.isnan(signalspaceinterpnorm[:,iiii])).any() == True:
                #print('signalspaceinterp[:,iiii] is ',*signalspaceinterp[:,iiii])
                #print('signalspaceinterp.max() is ',signalspaceinterp.max())
                #print('Found a NaN!!!! @ value ',iiii,' which corresponds to rho values of ', self.xrho[iiii])
            signalspacetimeinterp[:,iiii] = np.interp(x,signaltime,signalspaceinterpnorm[:,iiii])
            for jj in range(len(x)):
                background[jj,iiii] = coeff[numdloc+numvloc]*x[jj] + coeff[numdloc+numvloc+1]
                signalwithbackground[jj,iiii] = signalspacetimeinterp[jj,iiii] + background[jj,iiii]

        signalwithbackground = signalwithbackground*coeff[numdloc+numvloc+2] #/ signalwithbackground.max() 

        signalwithbackground = signalwithbackground.flatten('F')

        signalcounter = 0


        '''
        signalFe24arraysumspatial = np.sum(signalFe24,axis=1)
        yFe24 = signalFe24arraysumspatial*coeff[numdloc+numvloc+5+(3*signalcounter)]*10**8.0 
        backgroundFe24 = np.zeros(len(self.xxics))
        ycalFe24 = np.interp(self.xxics,signaltime,yFe24)

        backgroundFe24 = np.zeros(len(self.xxics))
        ytotFe24 = np.zeros(len(self.xxics))
        for ii in range(len(self.xxics)):
            backgroundFe24[ii] = coeff[numdloc+numvloc+3+(3*signalcounter)]*self.xxics[ii] + coeff[numdloc+numvloc+4+(3*signalcounter)]
            ytotFe24[ii] = ycalFe24[ii] + backgroundFe24[ii] 

        signalwithbackground = np.append(signalwithbackground,ytotFe24)
        signalcounter = signalcounter+1
        '''

        
        signalFe22arraysumspatial = np.sum(signalFe22,axis=1)
        yFe22 = signalFe22arraysumspatial*coeff[numdloc+numvloc+5+(3*signalcounter)]*10**8.0 
        backgroundFe22 = np.zeros(len(self.xhexos))
        ycalFe22 = np.interp(self.xhexos,signaltime,yFe22)

        backgroundFe22 = np.zeros(len(self.xhexos))
        ytotFe22 = np.zeros(len(self.xhexos))
        for ii in range(len(self.xhexos)):
            backgroundFe22[ii] = coeff[numdloc+numvloc+3+(3*signalcounter)]*self.xhexos[ii] + coeff[numdloc+numvloc+4+(3*signalcounter)]
            ytotFe22[ii] = ycalFe22[ii] + backgroundFe22[ii]


        signalwithbackground = np.append(signalwithbackground,ytotFe22)
        signalcounter = signalcounter+1
        

        '''
        signalFe21arraysumspatial = np.sum(signalFe21,axis=1)
        yFe21 = signalFe21arraysumspatial*coeff[numdloc+numvloc+5+(3*signalcounter)]*10**8.0 
        backgroundFe21 = np.zeros(len(self.xhexos))
        ycalFe21 = np.interp(self.xhexos,signaltime,yFe21)

        backgroundFe21 = np.zeros(len(self.xhexos))
        ytotFe21 = np.zeros(len(self.xhexos))
        for ii in range(len(self.xhexos)):
            backgroundFe21[ii] = coeff[numdloc+numvloc+3+(3*signalcounter)]*self.xhexos[ii] + coeff[numdloc+numvloc+4+(3*signalcounter)]
            ytotFe21[ii] = ycalFe21[ii] + backgroundFe21[ii]


        signalwithbackground = np.append(signalwithbackground,ytotFe21)
        signalcounter = signalcounter+1
        '''


        '''
        signalFe19arraysumspatial = np.sum(signalFe19,axis=1)
        yFe19 = signalFe19arraysumspatial*coeff[numdloc+numvloc+5+(3*signalcounter)]*10**8.0 
        backgroundFe19 = np.zeros(len(self.xhexos))
        ycalFe19 = np.interp(self.xhexos,signaltime,yFe19)

        backgroundFe19 = np.zeros(len(self.xhexos))
        ytotFe19 = np.zeros(len(self.xhexos))
        for ii in range(len(self.xhexos)):
            backgroundFe19[ii] = coeff[numdloc+numvloc+3+(3*signalcounter)]*self.xhexos[ii] + coeff[numdloc+numvloc+4+(3*signalcounter)]
            ytotFe19[ii] = ycalFe19[ii] + backgroundFe19[ii]
 

        signalwithbackground = np.append(signalwithbackground,ytotFe19)
        signalcounter = signalcounter+1



        signalFe18arraysumspatial = np.sum(signalFe18,axis=1)
        yFe18 = signalFe18arraysumspatial*coeff[numdloc+numvloc+5+(3*signalcounter)]*10**8.0 
        backgroundFe18 = np.zeros(len(self.xhexos))
        ycalFe18 = np.interp(self.xhexos,signaltime,yFe18)

        backgroundFe18 = np.zeros(len(self.xhexos))
        ytotFe18 = np.zeros(len(self.xhexos))
        for ii in range(len(self.xhexos)):
            backgroundFe18[ii] = coeff[numdloc+numvloc+3+(3*signalcounter)]*self.xhexos[ii] + coeff[numdloc+numvloc+4+(3*signalcounter)]
            ytotFe18[ii] = ycalFe18[ii] + backgroundFe18[ii]


        signalwithbackground = np.append(signalwithbackground,ytotFe18)
        signalcounter = signalcounter+1
        '''

        '''
        signalFe12arraysumspatial = np.sum(signalFe12,axis=1)
        yFe12 = signalFe12arraysumspatial*coeff[numdloc+numvloc+5+(3*signalcounter)]*10**8.0 
        backgroundFe12 = np.zeros(len(self.xhexos))
        ycalFe12 = np.interp(self.xhexos,signaltime,yFe12)

        backgroundFe12 = np.zeros(len(self.xhexos))
        ytotFe12 = np.zeros(len(self.xhexos))
        for ii in range(len(self.xhexos)):
            backgroundFe12[ii] = coeff[numdloc+numvloc+3+(3*signalcounter)]*self.xhexos[ii] + coeff[numdloc+numvloc+4+(3*signalcounter)]
            ytotFe12[ii] = ycalFe12[ii] + backgroundFe12[ii]


        signalwithbackground = np.append(signalwithbackground,ytotFe12)
        signalcounter = signalcounter+1
        '''

        signalcounter = 0
        

        return {'signal':signalwithbackground
                ,'totalname':strahlexecution['totalname']
                ,'rhodval':strahlexecution['rhodval']
                ,'rhovval':strahlexecution['rhovval']}
    
    def residual(self
                 ,coeff
                 ,x=None
                 ,numdloc=None
                 ,choiceedgerhod=None
                 ,choiceaxisrhod=None
                 ,numvloc=None
                 ,choiceedgerhov=None
                 ,choiceaxisrhov=None):
        """
        Return the residual for the normalized impurity density.
        This method can be bassed to mir_mpfit.
        """
        
        yfit = (self.strahlmodelsignal(coeff=coeff
                                       ,x=x
                                       ,numdloc=numdloc
                                       ,choiceedgerhod=choiceedgerhod
                                       ,choiceaxisrhod=choiceaxisrhod
                                       ,numvloc=numdloc
                                       ,choiceedgerhov=choiceedgerhov
                                       ,choiceaxisrhov=choiceaxisrhov))['signal']
        

        #print('yfit.shape ',yfit.shape)
        #print('self.data.shape ',(self.data).shape)
        #print('self.sigma.shape ',(self.sigma).shape)

        residual = (self.data - yfit) / self.sigma
        
          
        return {'residual':residual.flatten('F')}
    
    def strahlfit(self
                  ,x=None
                  ,numdloc=None
                  ,choiceedgerhod=None
                  ,choiceaxisrhod=None
                  ,numvloc=None
                  ,choiceedgerhov=None
                  ,choiceaxisrhov=None):
        
        residual_keywords = {'x':x
                             ,'numdloc':numdloc
                             ,'choiceedgerhod':choiceedgerhod
                             ,'choiceaxisrhod':choiceaxisrhod
                             ,'numvloc':numvloc
                             ,'choiceedgerhov':choiceedgerhov
                             ,'choiceaxisrhov':choiceaxisrhov}

        parameterdvalues = np.zeros(residual_keywords['numdloc'])
        parametervvalues = np.zeros(residual_keywords['numvloc'])
        
        parameterdvalues[0] = 0.05
        parameterdvalues[1] = 0.05
        parameterdvalues[2] = 0.5
        parameterdvalues[3] = 1.0
        #parameterdvalues[4] = 2.0
        #parameterdvalues[5] = 2.0
        #parameterdvalues[6] = 5.0
        #parameterdvalues[7] = 5.0

        parametervvalues[0] = 0.0
        parametervvalues[1] = -0.1
        parametervvalues[2] = 0.1
        parametervvalues[3] = -1.0
        #parametervvalues[4] = -2.0
        #parametervvalues[5] = -0.5
        #parametervvalues[6] = -10.0
        #parametervvalues[7] = -10.0

        parametertransport = np.append(parameterdvalues,parametervvalues) 

        parameter_guess = np.array([0.0
                                    ,0.0
                                    ,8.0
                                    #,-3500.0 #Fe24
                                    #,18000.0
                                    #,1.57
                                    ,10.00 #Fe22
                                    ,20000.0
                                    ,0.5])
                                    #,10.00 #Fe21
                                    #,4000.0
                                    #,0.5
                                    #,10.00 #Fe19
                                    #,1500.0
                                    #,0.3
                                    #,10.00 #Fe18
                                    #,1500.0
                                    #,0.2])
                                    #,10.00 #Fe12
                                    #,1500.0
                                    #,0.1])
        

        parameter_guess = np.append(parametertransport,parameter_guess)
        #print('parameter_guess ',*parameter_guess)
        #ionstage = 24
        
        parinfo=[{'step':0.0001,'fixed':True,'limited':[0,0],'limits':[0.0,0.0]} for jj in range(len(parameter_guess))]   

        
        for ii in range(len(parameter_guess)):
            if ii < len(parameterdvalues):
                parinfo[ii]['limited'][0] = 1
            if (ii == len(parameterdvalues) and choiceaxisrhov == 0.0) or ii == len(parametertransport) or ii == len(parametertransport)+1:
                parinfo[ii]['fixed'] = True
            #if ii == len(parametertransport)+6 or ii == len(parametertransport)+9 or ii == len(parametertransport)+12 or ii == len(parametertransport)+15 or ii == len(parametertransport)+18:
                #parinfo[ii]['fixed'] = True
            else:
                parinfo[ii]['fixed'] = False
        
            
        parinfoarray = np.zeros(len(parinfo))
        for ii in range(len(parinfo)):
            parinfoarray[ii] = parinfo[ii]['step']

        
        mp = mpfit.mpfit(self.residual
                         ,parameter_guess
                         ,parinfo=parinfo
                         ,residual_keywords=residual_keywords
                         ,maxiter=500
                         ,quiet=True)
        
        
        logging.info(mp.errmsg)
        logging.info('mpfit status:')
        logging.info(mp.statusString())
        
        if mp.perror is None: mp.perror = parameter_guess*0
        

        # Useful for debugging.
        #yfit_guess = self.strahlmodelsignal(parameter_guess,x=x,numdloc=numdloc,numvloc=numvloc)

                  
        yfit = self.strahlmodelsignal(coeff=mp.params
                                      ,x=x
                                      ,numdloc=numdloc
                                      ,choiceedgerhod=choiceedgerhod
                                      ,choiceaxisrhod=choiceaxisrhod
                                      ,numvloc=numvloc
                                      ,choiceedgerhov=choiceedgerhov
                                      ,choiceaxisrhov=choiceaxisrhov)


        #print(mp.perror)
        #print(mp.fnorm)
        print('The fit status is ', mp.status)
        print('The number of iterations completed: ', mp.niter)
        print('The number of calls to strahl: ', mp.nfev)
        
    
        return {'type':'Direct matching with strahl'
                ,'netcdffilename': yfit['totalname']
                ,'summedchisquared':mp.fnorm
                ,'fit':yfit['signal']
                ,'rhodval':yfit['rhodval']
                ,'rhovval':yfit['rhovval']
                ,'params':mp.params
                ,'perror':mp.perror
                ,'coeff':parameter_guess
                ,'minparamstepsize': parinfoarray}
        
    def print_results(self, results, flag=None):
        print('Results:')
        print('{:>5s}  {:>6s}  {:>4s}  {:>15s}  {:>15s}'.format(
            'param'
            ,'start value'
            ,'fit'
            ,'error'
            ,'minstepsize'))
        
        for ii in range(len(results['params'])):
            print('{:>5d}  {:>10.4f}  {:>6.8f} +/- {:>4.8f}  {:>4.8f}'.format(
                ii
                ,results['coeff'][ii]
                ,results['params'][ii]
                ,results['perror'][ii]
                ,results['minparamstepsize'][ii]))
        print('\n','The total chisquared value is: ', results['summedchisquared'],'\n')
        if flag == None:
            print('\n','The result netcdf filename is: ', results['netcdffilename'],'\n')
        
        
            
    def plot_results(self, results):
        # Plot the fit results
        pyplot.close()

        #print('self.x.shape ',self.x.shape)
        #print('self.xrho.shape ',self.xrho.shape)

        lengthbeforetile = len(self.x)
        #print('lengthbeforetile ',lengthbeforetile)
        self.x = np.tile(self.x,len(self.xrho))

        lengthbeforesumsignal = len(self.x)
        #self.x = np.append(self.x,self.xxics)

        dimension, data, attribute = readstrahlnetcdf(results['netcdffilename'])

        rho = data['rho_poloidal_grid']
        radius = data['radius_grid']*.01

        #The rows of diff and convv are just different timeslices
        #The diff and convv are assumed constant in time hence the 0
        #The values output from STRAHL are in cm and need to be converted into m
        diff = data['anomal_diffusion'][0,:]*(0.01**2)
        convv = data['anomal_drift'][0,:]*0.01
        
        for kk in range(len(self.xrho)):
            if kk == 0:
                dataarray = (np.expand_dims(self.data[0:lengthbeforetile],axis=0)).T
                sigmadataarray = (np.expand_dims(self.sigma[0:lengthbeforetile],axis=0)).T
                STRAHLdataarray = (np.expand_dims(results['fit'][0:lengthbeforetile],axis=0)).T
                #print('dataarray.shape ',dataarray.shape)
            elif kk > 0:
                dataarray = np.append(dataarray,(np.expand_dims(self.data[kk*lengthbeforetile:(kk+1)*lengthbeforetile],axis=0)).T,axis=1)
                sigmadataarray = np.append(sigmadataarray,(np.expand_dims(self.sigma[kk*lengthbeforetile:(kk+1)*lengthbeforetile],axis=0)).T,axis=1)
                STRAHLdataarray = np.append(STRAHLdataarray,(np.expand_dims(results['fit'][kk*lengthbeforetile:(kk+1)*lengthbeforetile],axis=0)).T,axis=1)
                    
        #print('dataarray.shape ',dataarray.shape)


        #print('np.array([self.xrho]*lengthbeforetile).shape ',np.array([self.xrho]*lengthbeforetile).shape)
        #print('np.array([self.x[:lengthbeforetile]]*len(self.xrho)).shape ',np.array([self.x[:lengthbeforetile]]*len(self.xrho)).shape)
        #print('dataarray.flatten().shape ', (dataarray.flatten()).shape)

        fsize=(10.2,8)
        linewidth = 160
        startstr = 'Start: '+(np.array_str(results['coeff'],max_line_width=linewidth))[1:-1]
        resultstr = 'Result: '+(np.array_str(results['params'],max_line_width=linewidth))[1:-1]
        perrorstr = 'Perror: '+(np.array_str(results['perror'],max_line_width=linewidth))[1:-1]
        stepstr = 'Step: '+(np.array_str(results['minparamstepsize'],max_line_width=linewidth))[1:-1]
        rhodvalstr = 'Rho values: '+(np.array_str(results['rhodval'],max_line_width=linewidth))[1:-1]

        title = startstr+'\n'+resultstr+'\n'+perrorstr+'\n'+rhodvalstr

        '''
        pyplot.figure(0,figsize=fsize)
        for ii in range(len(self.xrho)):
            startindex = ii*lengthbeforetile
            endindex = ii*lengthbeforetile+lengthbeforetile
            pyplot.errorbar(self.x[:lengthbeforetile],self.data[startindex:endindex],yerr=self.sigma[startindex:endindex])
            pyplot.scatter(self.x[:lengthbeforetile], results['fit'][startindex:endindex])
        pyplot.xlabel("Time (s)",fontsize=15)
        pyplot.ylabel("Intensity (arb)",fontsize=15)
        pyplot.title('Start: '+str(results['coeff'])[1:-1]+'\n'+'Result: '+str(results['params'])[1:-1]+'\n'+'perror: '+str(results['perror'])[1:-1]+'\n'+'Step: '+str(results['minparamstepsize'])[1:-1],fontsize=15)

        pyplot.figure(1)
        pyplot.errorbar(self.x[:lengthbeforesumsignal],self.data[:lengthbeforesumsignal],yerr=self.sigma[:lengthbeforesumsignal], fmt='bo', ms=2)
        #pyplot.scatter(self.x,self.data)
        pyplot.scatter(self.x[:lengthbeforesumsignal], results['fit'][:lengthbeforesumsignal], color='red')
        pyplot.xlabel("Time (s)")
        pyplot.ylabel("Intensity (arb)")
        pyplot.title('Start: '+str(results['coeff'])[1:-1]+'\n'+'Result: '+str(results['params'])[1:-1]+'\n'+'perror: '+str(results['perror'])[1:-1]+'\n'+'Step: '+str(results['minparamstepsize'])[1:-1])
        '''

        '''
        fig = pyplot.figure(1,figsize=fsize)
        gs = gridspec.GridSpec(2,1,height_ratios=[2,1])
    
        ax1 = pyplot.subplot(gs[0])
    
        fig.suptitle(title,fontsize=15)
        ax1.errorbar(self.xxics,self.data[lengthbeforesumsignal:len(self.xxics)],yerr=self.sigma[lengthbeforesumsignal:len(self.xxics)], fmt='bo', ms=2)
        ax1.scatter(self.xxics, results['fit'][lengthbeforesumsignal:len(self.xxics)], color='red')
        ax1.set_ylabel('Intensity (arb)',fontsize=15)
        pyplot.setp(ax1.get_xticklabels(), visible=False)
    
        ax2 = pyplot.subplot(gs[1],sharex=ax1)
        ax2.scatter(self.xxics,(self.data[lengthbeforesumsignal:len(self.xxics)]-results['fit'][lengthbeforesumsignal:len(self.xxics)]))
        ax2.set_ylabel('Residual',fontsize=15)
        ax2.set_xlabel('Time (s)',fontsize=15)
        pyplot.subplots_adjust(hspace=.0)
        '''

        '''
        fig2 = pyplot.figure(2,figsize=fsize)
        gs2 = gridspec.GridSpec(2,1,height_ratios=[2,1])
    
        ax12 = pyplot.subplot(gs2[0])
    
        fig2.suptitle(title,fontsize=15)
        ax12.errorbar(self.xxics,self.data[lengthbeforesumsignal:lengthbeforesumsignal+len(self.xxics)],yerr=self.sigma[lengthbeforesumsignal:lengthbeforesumsignal+len(self.xxics)], fmt='bo', ms=2)
        ax12.scatter(self.xxics, results['fit'][lengthbeforesumsignal:lengthbeforesumsignal+len(self.xxics)], color='red')
        ax12.set_ylabel('Intensity (arb)',fontsize=15)
        pyplot.setp(ax12.get_xticklabels(), visible=False)
    
        ax22 = pyplot.subplot(gs2[1],sharex=ax12)
        ax22.scatter(self.xxics,self.sigma[lengthbeforesumsignal:lengthbeforesumsignal+len(self.xxics)]*(self.data[lengthbeforesumsignal:lengthbeforesumsignal+len(self.xxics)]-results['fit'][lengthbeforesumsignal:lengthbeforesumsignal+len(self.xxics)]))
        ax22.set_ylabel('Weighted residual',fontsize=15)
        ax22.set_xlabel('Time (s)',fontsize=15)
        pyplot.subplots_adjust(hspace=.0)
        '''

        
        
        figFe22 = pyplot.figure(22,figsize=fsize)
        gsFe22 = gridspec.GridSpec(2,1,height_ratios=[2,1])
    
        ax1Fe22 = pyplot.subplot(gsFe22[0])
    
        length = lengthbeforesumsignal#+len(self.xxics)

        figFe22.suptitle(title,fontsize=15)
        ax1Fe22.errorbar(self.xhexos,self.data[length:length+len(self.xhexos)],yerr=self.sigma[length:length+len(self.xhexos)], fmt='bo', ms=2)
        ax1Fe22.scatter(self.xhexos, results['fit'][length:length+len(self.xhexos)], color='red')
        ax1Fe22.set_ylabel('Intensity (arb)',fontsize=15)
        pyplot.setp(ax1Fe22.get_xticklabels(), visible=False)
    
        ax2Fe22 = pyplot.subplot(gsFe22[1],sharex=ax1Fe22)
        ax2Fe22.scatter(self.xhexos,self.sigma[length:length+len(self.xhexos)]*(self.data[length:length+len(self.xhexos)]-results['fit'][length:length+len(self.xhexos)]))
        ax2Fe22.set_ylabel('Weighted residual',fontsize=15)
        ax2Fe22.set_xlabel('Time (s)',fontsize=15)
        pyplot.subplots_adjust(hspace=.0)
        

        '''
        figFe21 = pyplot.figure(21,figsize=fsize)
        gsFe21 = gridspec.GridSpec(2,1,height_ratios=[2,1])
        ax1Fe21 = pyplot.subplot(gsFe21[0])
    
        length = lengthbeforesumsignal+len(self.xhexos) #+len(self.xxics)+len(self.xhexos)

        figFe21.suptitle(title,fontsize=15)
        ax1Fe21.errorbar(self.xhexos,self.data[length:length+len(self.xhexos)],yerr=self.sigma[length:length+len(self.xhexos)], fmt='bo', ms=2)
        ax1Fe21.scatter(self.xhexos, results['fit'][length:length+len(self.xhexos)], color='red')
        ax1Fe21.set_ylabel('Intensity (arb)',fontsize=15)
        pyplot.setp(ax1Fe21.get_xticklabels(), visible=False)
    
        ax2Fe21 = pyplot.subplot(gsFe21[1],sharex=ax1Fe21)
        ax2Fe21.scatter(self.xhexos,self.sigma[length:length+len(self.xhexos)]*(self.data[length:length+len(self.xhexos)]-results['fit'][length:length+len(self.xhexos)]))
        ax2Fe21.set_ylabel('Weighted residual',fontsize=15)
        ax2Fe21.set_xlabel('Time (s)',fontsize=15)
        pyplot.subplots_adjust(hspace=.0)
        '''

        '''
        figFe19 = pyplot.figure(19,figsize=fsize)
        gsFe19 = gridspec.GridSpec(2,1,height_ratios=[2,1])
        ax1Fe19 = pyplot.subplot(gsFe19[0])

        length = lengthbeforesumsignal+len(self.xhexos) #+len(self.xxics)+2*len(self.xhexos)

        figFe19.suptitle(title,fontsize=15)
        ax1Fe19.errorbar(self.xhexos,self.data[length:length+len(self.xhexos)],yerr=self.sigma[length:length+len(self.xhexos)], fmt='bo', ms=2)
        ax1Fe19.scatter(self.xhexos, results['fit'][length:length+len(self.xhexos)], color='red')
        ax1Fe19.set_ylabel('Intensity (arb)',fontsize=15)
        pyplot.setp(ax1Fe19.get_xticklabels(), visible=False)
    
        ax2Fe19 = pyplot.subplot(gsFe19[1],sharex=ax1Fe19)
        ax2Fe19.scatter(self.xhexos,self.sigma[length:length+len(self.xhexos)]*(self.data[length:length+len(self.xhexos)]-results['fit'][length:length+len(self.xhexos)]))
        ax2Fe19.set_ylabel('Weighted residual',fontsize=15)
        ax2Fe19.set_xlabel('Time (s)',fontsize=15)
        pyplot.subplots_adjust(hspace=.0)


        figFe18 = pyplot.figure(18,figsize=fsize)
        gsFe18 = gridspec.GridSpec(2,1,height_ratios=[2,1])
        ax1Fe18 = pyplot.subplot(gsFe18[0])

        length = lengthbeforesumsignal+2*len(self.xhexos) #+len(self.xxics)+3*len(self.xhexos)-len(self.xxics)

        figFe18.suptitle(title,fontsize=15)
        ax1Fe18.errorbar(self.xhexos,self.data[length:length+len(self.xhexos)],yerr=self.sigma[length:length+len(self.xhexos)], fmt='bo', ms=2)
        ax1Fe18.scatter(self.xhexos, results['fit'][length:length+len(self.xhexos)], color='red')
        ax1Fe18.set_ylabel('Intensity (arb)',fontsize=15)
        pyplot.setp(ax1Fe18.get_xticklabels(), visible=False)
    
        ax2Fe18 = pyplot.subplot(gsFe18[1],sharex=ax1Fe18)
        ax2Fe18.scatter(self.xhexos,self.sigma[length:length+len(self.xhexos)]*(self.data[length:length+len(self.xhexos)]-results['fit'][length:length+len(self.xhexos)]))
        ax2Fe18.set_ylabel('Weighted residual',fontsize=15)
        ax2Fe18.set_xlabel('Time (s)',fontsize=15)
        pyplot.subplots_adjust(hspace=.0)
        '''

        '''
        figFe12 = pyplot.figure(12,figsize=fsize)
        gsFe12 = gridspec.GridSpec(2,1,height_ratios=[2,1])
        ax1Fe12 = pyplot.subplot(gsFe12[0])

        length = lengthbeforesumsignal+len(self.xxics)+4*len(self.xhexos)

        figFe12.suptitle(title,fontsize=15)
        ax1Fe12.errorbar(self.xhexos,self.data[length:length+len(self.xhexos)],yerr=self.sigma[length:length+len(self.xhexos)], fmt='bo', ms=2)
        ax1Fe12.scatter(self.xhexos, results['fit'][length:length+len(self.xhexos)], color='red')
        ax1Fe12.set_ylabel('Intensity (arb)',fontsize=15)
        pyplot.setp(ax1Fe12.get_xticklabels(), visible=False)
    
        ax2Fe12 = pyplot.subplot(gsFe12[1],sharex=ax1Fe12)
        ax2Fe12.scatter(self.xhexos,self.sigma[length:length+len(self.xhexos)]*(self.data[length:length+len(self.xhexos)]-results['fit'][length:length+len(self.xhexos)]))
        ax2Fe12.set_ylabel('Weighted residual',fontsize=15)
        ax2Fe12.set_xlabel('Time (s)',fontsize=15)
        pyplot.subplots_adjust(hspace=.0)
        '''

        pyplot.figure(3,figsize=fsize)
        pyplot.errorbar(rho, diff, fmt='bo', ms=8)
        pyplot.errorbar(results['rhodval'], results['params'][:len(results['rhodval'])],yerr=results['perror'][:len(results['rhodval'])], fmt='ro', ms=7)
        pyplot.ylim(0,np.max(diff)*1.5)
        pyplot.xlabel('Rho (r/a)',fontsize=15)
        pyplot.ylabel('Diffusion coefficient (m^2/s)',fontsize=15)
        pyplot.title(title,fontsize=15)

        pyplot.figure(4,figsize=fsize)
        pyplot.errorbar(rho, convv, fmt='bo', ms=8)
        pyplot.errorbar(results['rhovval'], results['params'][len(results['rhodval']):len(results['rhodval'])+len(results['rhovval'])],yerr=results['perror'][len(results['rhodval']):len(results['rhodval'])+len(results['rhovval'])], fmt='ro', ms=7)
        pyplot.xlabel('Rho (r/a)',fontsize=15)
        pyplot.ylabel('Convective velocity (m/s)',fontsize=15)
        pyplot.title(title,fontsize=15)

        pyplot.figure(5,figsize=fsize)
        pyplot.contourf(self.xrho,self.x[:lengthbeforetile],dataarray)
        colorbar=pyplot.colorbar()
        colorbar.set_label('Intensity (arb)',fontsize=15)
        pyplot.xlabel("Rho (r/a)",fontsize=15)
        pyplot.ylabel("Time (s)",fontsize=15)
        pyplot.title(title,fontsize=15)

        '''
        pyplot.figure(6,figsize=fsize)
        pyplot.scatter((np.array([self.xrho]*lengthbeforetile)).flatten(),(np.array([self.x[:lengthbeforetile]]*len(self.xrho)).T).flatten(),c=dataarray.flatten(),s=16)
        colorbar=pyplot.colorbar()
        colorbar.set_label('Intensity (arb)',fontsize=15)
        pyplot.xlabel("Rho (r/a)",fontsize=15)
        pyplot.ylabel("Time (s)",fontsize=15)
        pyplot.title(title,fontsize=15)
        '''

        pyplot.figure(7,figsize=fsize)
        pyplot.contourf(self.xrho,self.x[:lengthbeforetile],STRAHLdataarray)
        colorbar=pyplot.colorbar()
        colorbar.set_label('Intensity (arb)',fontsize=15)
        pyplot.xlabel("Rho (r/a)",fontsize=15)
        pyplot.ylabel("Time (s)",fontsize=15)
        pyplot.title(title,fontsize=15)


        '''
        pyplot.figure(8,figsize=fsize)
        pyplot.scatter((np.array([self.xrho]*lengthbeforetile)).flatten(),(np.array([self.x[:lengthbeforetile]]*len(self.xrho)).T).flatten(),c=STRAHLdataarray.flatten(),s=16)
        colorbar=pyplot.colorbar()
        colorbar.set_label('Intensity (arb)',fontsize=15)
        pyplot.xlabel("Rho (r/a)",fontsize=15)
        pyplot.ylabel("Time (s)",fontsize=15)
        pyplot.title(title,fontsize=15)
        '''

        pyplot.figure(9,figsize=fsize)
        pyplot.contourf(self.xrho,self.x[:lengthbeforetile],STRAHLdataarray-dataarray)
        colorbar=pyplot.colorbar()
        colorbar.set_label('Residual (arb)',fontsize=15)
        pyplot.xlabel("Rho (r/a)",fontsize=15)
        pyplot.ylabel("Time (s)",fontsize=15)
        pyplot.title(title,fontsize=15)

        pyplot.figure(10,figsize=fsize)
        pyplot.contourf(self.xrho,self.x[:lengthbeforetile],sigmadataarray*(STRAHLdataarray-dataarray))
        colorbar=pyplot.colorbar()
        colorbar.set_label('Weighted residual (arb)',fontsize=15)
        pyplot.xlabel("Rho (r/a)",fontsize=15)
        pyplot.ylabel("Time (s)",fontsize=15)
        pyplot.title(title,fontsize=15)

        pyplot.show()

    def runanalysis(self,name=None,ionstage=None,signalflag=None):
        """
        Run the the fittering routine 

        This will fit the given data set and display the results.
        """

        dimension, data, attribute = readstrahlnetcdf(name)
        
        '''
        h = 6.626*10**-34
        c = 3*10**8
        wavelength = 1.8*10**-10
        '''
        
        self.ionstage=24
        
        if signalflag == False:
            #This is for matching to impurity density
            signal = data['impurity_density'][:,int(self.ionstage)]
            print('density is in use')
        elif signalflag == True:
            #This is for matching to impurity line radiation (where 24+ is listed in the 2nd spot in Fe.atomdat)
            #Units are W*cm^-3 so to get a better sigma elavualtion we will divide hc/lamdba
            signal = data['diag_lines_radiation'][:,1]
            print('radiation is in use')
        

        signalsumspatial = np.sum(signal,axis=1)
        timearray = data['time']
        
        numoftimeslices = len(timearray)      
        totaltimeslicesforloglinearfit = int(100)
        timesliceoffsetfrommax = int(30)
            
        timeindexmaxvalue = (np.where(signalsumspatial==signalsumspatial.max()))[0]
        
        if signalsumspatial.max()==0.0:
            timeindexmaxvalue=0
            
        timewindowforloglinearfit = np.arange(int(timeindexmaxvalue+timesliceoffsetfrommax)
                                                ,int(timeindexmaxvalue+timesliceoffsetfrommax+totaltimeslicesforloglinearfit)
                                                ,1)
        
        if timewindowforloglinearfit[-1] > numoftimeslices:
            raise Exception('Window is incorrect!!')
        
        '''
        print('timeindexmaxvalue is: ', timeindexmaxvalue)
        print('time at maxvalue is: ', timearray[timeindexmaxvalue])
        print('timewindowforloglinearfit',timewindowforloglinearfit)
        print('x_offset is: ', timearray[timewindowforloglinearfit[0]])
        print('x_offset.shape is: ', timearray[timewindowforloglinearfit[0]].shape)
        print('x2 is: ', timearray[timewindowforloglinearfit[:]])
        print('x2.shape is: ', timearray[timewindowforloglinearfit[:]].shape)
        
        print('signalsumspatial[timewindowforloglinearfit[:]] is: ', signalsumspatial[timewindowforloglinearfit[:]])
        print('data2 is: ', np.log(signalsumspatial[timewindowforloglinearfit[:]]))
        print('data2.shape is: ', np.log(signalsumspatial[timewindowforloglinearfit[:]]).shape)
        print('sigma2 is: ', np.sqrt(abs(signalsumspatial[timewindowforloglinearfit[:]])))
        print('sigma2.shape is: ', np.sqrt(abs(signalsumspatial[timewindowforloglinearfit[:]])).shape)
        print('signalsumspatial is : ', signalsumspatial)
        print('signalsumspatial.shape is : ', signalsumspatial.shape)
        '''

        #pyplot.plot(timearray[timewindowforloglinearfit[:]],signalsumspatial[timewindowforloglinearfit[:]])
        #pyplot.show()

        #pyplot.plot(timearray[timewindowforloglinearfit[:]],np.log(signalsumspatial[timewindowforloglinearfit[:]]))
        #pyplot.show()
            
        self.x_offset = timearray[timewindowforloglinearfit[0]]
        self.x2 = timearray[timewindowforloglinearfit[:]]
        self.data2 = np.log(signalsumspatial[timewindowforloglinearfit[:]])
        self.sigma2 = 1#np.sqrt(abs(signalsumspatial[timewindowforloglinearfit[:]])) / signalsumspatial[timewindowforloglinearfit[:]].max()
        transportresults = self.loglinearfit()
        
        #print('self.sigma2 is: ', self.sigma2)
        #print('self.sigma is: ',self.sigma)
        
        self.print_results(transportresults, flag=False)
        #self.plot_results(transportresults)
        
        timeindexforstartofrise = (np.argwhere(signalsumspatial>1e-8))
        print('timeindexforstartofrise[0] is: ', *timeindexforstartofrise[0])
        print('timeindexmaxvalue[0] is: ', timeindexmaxvalue[0])
        print('timearray[timeindexmaxvalue[0]] is: ',timearray[timeindexmaxvalue[0]])
        print('Time from neutral LBO injection to max value is: ',timearray[timeindexmaxvalue[0]]-2.46)
        print('Time from signal rise to maxvalue is: ', *timearray[timeindexmaxvalue[0]]-timearray[timeindexforstartofrise[0]])
        print('The lambda parameter is: ', transportresults['params'][1])
        print('Tau in milliseconds is: ', 1000.0/(transportresults['params'][1]))


        startstr = 'Start: '+(np.array_str(transportresults['coeff'],max_line_width=200))[1:-1]
        resultstr = 'Result: '+(np.array_str(transportresults['params'],max_line_width=200))[1:-1]
        perrorstr = 'Perror: '+(np.array_str(transportresults['perror'],max_line_width=200))[1:-1]
        stepstr = 'Step: '+(np.array_str(transportresults['minparamstepsize'],max_line_width=200))[1:-1]

        title = startstr+'\n'+resultstr+'\n'+perrorstr

        fsize=(10.2,8)
        pyplot.figure(11,figsize=fsize)
        #pyplot.errorbar(self.x2,np.exp(self.data2),yerr=self.sigma2, fmt='bo',ms=2)
        pyplot.errorbar(self.x2,np.exp(self.data2), fmt='bo',ms=8)
        pyplot.errorbar(self.x2, np.exp(transportresults['fit']),fmt='ro',ms=10)#,yerr=np.exp(transportresults['perror']),fmt='ro',ms=10)
        pyplot.xlabel("Time (s)",fontsize=15)
        pyplot.ylabel("Intensity (arb)",fontsize=15)
        pyplot.title(title+'\n'+'Tau in milliseconds is: '+ str(1000.0/(transportresults['params'][1]))+'\n'+name,fontsize=15)
        
        
        pyplot.figure(12,figsize=fsize)   
        pyplot.errorbar(timearray[np.arange(numoftimeslices)[:]],signalsumspatial, fmt='bo',ms=8)
        pyplot.errorbar(self.x2, np.exp(transportresults['fit']),fmt='ro',ms=10)#,yerr=np.exp(transportresults['perror']), fmt='ro',ms=10)
        pyplot.yscale('log')
        pyplot.ylim(1e-6,1e-3)
        pyplot.xlim(timearray[27],timearray[-1])
        pyplot.xlabel("Time (s)",fontsize=15)
        pyplot.ylabel("Intensity (arb)",fontsize=15)
        pyplot.title(title+'\n'+'Tau in milliseconds is: '+ str(1000.0/(transportresults['params'][1]))+'\n'+name,fontsize=15)
        pyplot.show()
        
        return {'netcdffilename': name
                ,'Ionstage':ionstage
                ,'Tau': 1000.0/(transportresults['params'][1])
                ,'timeinj':timearray[timeindexmaxvalue[0]]-2.46
                ,'timerise':(timearray[timeindexmaxvalue[0]]-timearray[timeindexforstartofrise[0]])[0]
                ,'timemaxvalueindex':timeindexmaxvalue[0]
                ,'timeindexstartrise':timeindexforstartofrise[0][0]}    
        
    def runstrahlgroup(self,ionstage):
        path = '/draco/u/petr/strahlnew_copy/result/op12a_171122022_FeLBO3_contour2/'
        #totfilerad = open('/draco/u/petr/strahlnew_copy/result/'+'op12a_171122022_FeLBO3data'+str(ionstage)+'radiation2','w')
        totfileden = open('/draco/u/petr/strahlnew_copy/result/'+'op12a_171122022_FeLBO3data'+str(ionstage)+'density2','w')
        #totfilerad.write('ionstage is: {} and this file is calculated using radiation time trace\n'.format(ionstage))
        totfileden.write('ionstage is: {} and this file is calculated using density time trace\n'.format(ionstage))
        #temprad=[]
        tempden=[]
        counter = 0
        for filename in os.listdir(path):
            dimension, data, attribute = readstrahlnetcdf(path+filename)
            diff2, v2 = data['anomal_diffusion'], data['anomal_drift']
            #diff = float(filename[1:3] + '.' + filename[4:6])
            #conv = float(filename[7:9] + '.' + filename[10:12])
            print('filename is: ',filename)
            print('diff2 is: ',0.0001*diff2[0,0])#,'  while diff is: ', diff)
            print('v2 is: ', 0.01*v2[0,1])#, ' while conv is: ', conv)
            try:
                #resultrad=self.runanalysis(path+filename,ionstage,signalflag=True)
                #temprad.append((0.0001*diff2[0,0],0.01*v2[0,1],resultrad['Tau'],resultrad['timeinj'],resultrad['timerise'],resultrad['timemaxvalueindex'],resultrad['timeindexstartrise']))
                #totfilerad.write('{} {} {} {} {} {} {} {}\n'.format(resultrad['filename'],0.0001*diff2[0,0],0.01*v2[0,1],resultrad['Tau'],resultrad['timeinj'],resultrad['timerise'],resultrad['timemaxvalueindex'],resultrad['timeindexstartrise']))
                
                resultden=self.runanalysis(path+filename,ionstage,signalflag=False)
                tempden.append((0.0001*diff2[0,0],0.01*v2[0,1],resultden['Tau'],resultden['timeinj'],resultden['timerise'],resultden['timemaxvalueindex'],resultden['timeindexstartrise']))
                totfileden.write('{} {} {} {} {} {} {} {}\n'.format(resultden['filename'],0.0001*diff2[0,0],0.01*v2[0,1],resultden['Tau'],resultden['timeinj'],resultden['timerise'],resultden['timemaxvalueindex'],resultden['timeindexstartrise']))
            except:
                counter=counter+1
                pass
        #totfilerad.close()
        totfileden.close()
        print('Counter is: ', counter)
        
        return np.asanyarray(tempden)#,np.asanyarray(tempden)]
    
    def runstrahlgroupcontourplot(self,ionstage,flag=False):
        if flag == False:
            data = self.runstrahlgroup(ionstage=ionstage)
            #datarad = data#[0]
            dataden = data#[1]
            #datarad0 = datarad[:,0]
            #datarad1 = datarad[:,1]
            #datarad2 = datarad[:,2]
            dataden0 = dataden[:,0]
            dataden1 = dataden[:,1]
            dataden2 = dataden[:,2]
        elif flag == True:
            testvalue = 1
            datarad = np.loadtxt('C:/Users/petr/Desktop/Peter_Python/op12a_171122022_FeLBOdata24radiation',delimiter=' ',skiprows=1,usecols=(1,2,3,4,5,6,7))
            dataden = np.loadtxt('C:/Users/petr/Desktop/Peter_Python/op12a_171122022_FeLBOdata24density',delimiter=' ',skiprows=1,usecols=(1,2,3,4,5,6,7))
            #print('data[:,0].shape',data[:,0].shape)
            #print('data[:,1].shape',data[:,1].shape)
            #print('data[:,2].shape',data[:,2].shape)
            datarad[datarad[:,0].argsort()]
            datarad[datarad[:,1].argsort()]
            dataden[dataden[:,0].argsort()]
            dataden[dataden[:,1].argsort()] 
            #print('testvalue is: ', testvalue)
            #print('data[::testvalue,0].shape',data[::testvalue,0].shape)
            #print('data[::testvalue,1].shape',data[::testvalue,1].shape)
            #print('data[::testvalue,2].shape',data[::testvalue,2].shape)
            datarad0 = datarad[::testvalue,0]
            datarad1 = datarad[::testvalue,1]
            datarad2 = datarad[::testvalue,2]
            #datarad2[datarad2.argsort()]
            print(datarad2.max())
            
            dataden0 = dataden[::testvalue,0]
            dataden1 = dataden[::testvalue,1]
            dataden2 = dataden[::testvalue,2]
            #dataden2[dataden2.argsort()]
            print(dataden2.max())

            
        #rbfrad = scipy.interpolate.Rbf(datarad0,datarad1,datarad2,function='linear')
        #xirad = np.linspace(datarad0.min(),datarad0.max(),100) 
        #yirad = np.linspace(datarad1.min(),datarad1.max(),100)
        #xirad2,yirad2 = np.meshgrid(xirad,yirad)
        #zirad2 = rbfrad(xirad2,yirad2)
        #print('radiation ready')

        rbfden = scipy.interpolate.Rbf(dataden0,dataden1,dataden2,function='linear')
        xiden = np.linspace(dataden0.min(),dataden0.max(),100) 
        yiden = np.linspace(dataden1.min(),dataden1.max(),100)
        xiden2,yiden2 = np.meshgrid(xiden,yiden)
        ziden2 = rbfden(xiden2,yiden2)
        '''
        pyplot.figure()
        pyplot.contourf(xirad2,yirad2,zirad2,np.arange(0,300,10),extend='both')
        pyplot.colorbar()
        pyplot.xlim(0,2.5)
        pyplot.ylim(-2.5,2.5)
        pyplot.xlabel('D (m^2/s)')
        pyplot.ylabel('v (m/s)')
        pyplot.title('Confinement times in ms for FeXXV from radiation')
        
        pyplot.figure()
        CS = pyplot.contour(xirad2,yirad2,zirad2,np.arange(0,300,10),extend='both')
        pyplot.colorbar()
        pyplot.clabel(CS,fontsize=10)
        pyplot.xlim(0,2.5)
        pyplot.ylim(-2.5,2.5)
        pyplot.xlabel('D (m^2/s)')
        pyplot.ylabel('v (m/s)')
        pyplot.title('Confinement times in ms for FeXXV from radiation')
        
        pyplot.figure()
        pyplot.scatter(datarad0,datarad1,c=datarad2)
        pyplot.colorbar()
        pyplot.xlim(0,2.5)
        pyplot.ylim(-2.5,2.5)
        pyplot.xlabel('D (m^2/s)')
        pyplot.ylabel('v (m/s)')
        pyplot.title('Confinement times in ms for FeXXV from radiation')
        '''
        pyplot.figure()
        pyplot.contourf(xiden2,yiden2,ziden2,np.arange(0,300,10),extend='both')
        pyplot.colorbar()
        pyplot.xlim(0,2.5)
        pyplot.ylim(-2.5,2.5)
        pyplot.xlabel('D (m^2/s)')
        pyplot.ylabel('v (m/s)')
        pyplot.title('Confinement times in ms for FeXXV from density')
        
        pyplot.figure()
        CSden = pyplot.contour(xiden2,yiden2,ziden2,np.arange(0,300,10),extend='both')
        pyplot.colorbar()
        pyplot.clabel(CSden,fontsize=10)
        pyplot.xlim(0,2.5)
        pyplot.ylim(-2.5,2.5)
        pyplot.xlabel('D (m^2/s)')
        pyplot.ylabel('v (m/s)')
        pyplot.title('Confinement times in ms for FeXXV from density')
        
        pyplot.figure()
        pyplot.scatter(dataden0,dataden1,c=dataden2)
        pyplot.colorbar()
        pyplot.xlim(0,2.5)
        pyplot.ylim(-2.5,2.5)
        pyplot.xlabel('D (m^2/s)')
        pyplot.ylabel('v (m/s)')
        pyplot.title('Confinement times in ms for FeXXV from density')
        
        pyplot.show()

    def runfit(self,test=True,name=None,sigma=None):
        
        sigma=sigma
        name=name
        if test == True:
            signal=np.loadtxt('/draco/u/petr/Peter_Python/'+name+'.csv',delimiter=',')
            time=np.loadtxt('/draco/u/petr/Peter_Python/'+name+'time.csv',delimiter=',')
            rho=np.loadtxt('/draco/u/petr/Peter_Python/'+name+'rho.csv',delimiter=',')
            #signal[signal < 0] = 0
            #totalsignal = np.zeros(len(signal))
            #normalizedsignal = signal / signal.max()
            #for ii in range(len(signal)):
            #    totalsignal[ii] = normalizedsignal[ii] + (-0.05 * time[ii]) + 0.4
            #totalsignal[totalsignal < 0] = 0
            timeindexmaxvalue = (np.where(signal/signal.max()==(signal/signal.max()).max()))[0]
            indexesbeforemax = timeindexmaxvalue#45
            indexesaftermax = -(timeindexmaxvalue+1)#125
            self.x = time[int(timeindexmaxvalue-indexesbeforemax):int(timeindexmaxvalue+indexesaftermax)]#*len(rho)
            self.xrho = rho
            #self.data = totalsignal[int(timeindexmaxvalue-indexesbeforemax):int(timeindexmaxvalue+indexesaftermax)] / (totalsignal[int(timeindexmaxvalue-indexesbeforemax):int(timeindexmaxvalue+indexesaftermax)]).max()
            #self.data =  signal[int(timeindexmaxvalue-indexesbeforemax):int(timeindexmaxvalue+indexesaftermax)] / (signal[int(timeindexmaxvalue-indexesbeforemax):int(timeindexmaxvalue+indexesaftermax)]).max()
            self.data =  signal[int(timeindexmaxvalue-indexesbeforemax):int(timeindexmaxvalue+indexesaftermax)] / (signal[int(timeindexmaxvalue-indexesbeforemax):int(timeindexmaxvalue+indexesaftermax)]).max()
            self.sigma = 0.01#np.sqrt(abs(signal[int(timeindexmaxvalue-indexesbeforemax):int(timeindexmaxvalue+indexesaftermax)])) / (signal[int(timeindexmaxvalue-indexesbeforemax):int(timeindexmaxvalue+indexesaftermax)]).max()
            self.ionstage = 24

        elif test == False:
            #name=171122022HRXISdata
            
            #name='w7x_ar17_171123036_xics_temp_inverted_WEmiss'
            signal=np.loadtxt('/draco/u/petr/Peter_Python/'+name+'.csv',delimiter=',')
            time=10**(-3.0)*np.loadtxt('/draco/u/petr/Peter_Python/'+name+'time.csv',delimiter=',')
            rho=np.loadtxt('/draco/u/petr/Peter_Python/'+name+'rho.csv',delimiter=',')
            sigma=np.loadtxt('/draco/u/petr/Peter_Python/'+name+'sigma.csv',delimiter=',')

            timeindexmaxvalue = (np.where(signal/signal.max()==(signal/signal.max()).max()))[0]
            indexesbeforemax = timeindexmaxvalue#50
            indexesaftermax = -timeindexmaxvalue-1#100
            

            #print('timeindexmaxvalue ',timeindexmaxvalue)
            #print('indexesbeforemax ',indexesbeforemax)
            #print('indexesaftermax ',indexesaftermax)
            #print('int(timeindexmaxvalue-indexesbeforemax) ',int(timeindexmaxvalue-indexesbeforemax))
            #print('int(timeindexmaxvalue+indexesaftermax) ',int(timeindexmaxvalue+indexesaftermax))


            startspatialindex = 0
            endspatialindex = -1
            
            signalsliced = signal[int(timeindexmaxvalue-indexesbeforemax):int(timeindexmaxvalue+indexesaftermax),startspatialindex:endspatialindex] #/ (signal[int(timeindexmaxvalue-indexesbeforemax):int(timeindexmaxvalue+indexesaftermax),startspatialindex:endspatialindex]).max()
            sigmasliced = sigma[int(timeindexmaxvalue-indexesbeforemax):int(timeindexmaxvalue+indexesaftermax),startspatialindex:endspatialindex]

            timesliced = time[int(timeindexmaxvalue-indexesbeforemax):int(timeindexmaxvalue+indexesaftermax)]
            rhosliced = rho[startspatialindex:endspatialindex]

            
            totnameFe24='171123036ar17data'            
            signalFe24spatialsum = np.loadtxt('/draco/u/petr/Peter_Python/'+totnameFe24+'.csv',delimiter=',')
            signalFe24spatialsumtime = np.loadtxt('/draco/u/petr/Peter_Python/'+totnameFe24+'time.csv',delimiter=',')

            indexesbeforemaxspatialsumxics = (np.abs(signalFe24spatialsumtime - timesliced[int(timeindexmaxvalue-indexesbeforemax)])).argmin()
            indexesaftermaxspatialsumxics = (np.abs(signalFe24spatialsumtime - timesliced[int(timeindexmaxvalue+indexesaftermax)])).argmin()

            #print('indexesbeforemaxspatialsum ',indexesbeforemaxspatialsum)             
            #print('indexesaftermaxspatialsum ',indexesaftermaxspatialsum)
            '''
            timeindexmaxvaluespatialsum = (np.where(signalspatialsum==signalspatialsum.max()))[0]
            indexesbeforemaxspatialsum = 50
            indexesaftermaxspatialsum = 100
            '''

            sigmascale = 1.0 #np.sqrt(1475)
            sigmaFe24spatialsum = sigmascale*np.sqrt(abs(signalFe24spatialsum[int(indexesbeforemaxspatialsumxics):int(indexesaftermaxspatialsumxics)])) #/ (signalspatialsum[int(indexesbeforemaxspatialsum):int(indexesaftermaxspatialsum)]).max()

            '''
            self.data = np.append(signalsliced.flatten('F'),signalFe24spatialsum[int(indexesbeforemaxspatialsumxics):int(indexesaftermaxspatialsumxics)])
            self.xxics = signalFe24spatialsumtime[int(indexesbeforemaxspatialsumxics):int(indexesaftermaxspatialsumxics)]
            self.sigma = np.append(sigmasliced.flatten('F'),sigmaFe24spatialsum)
            '''
            
            totnameFe22='20171123.036_signalFeXXIII_data_intensity_corrected_pixel_width_5'            
            signalFe22spatialsum = np.loadtxt('/draco/u/petr/Peter_Python/'+totnameFe22+'.csv',delimiter=',')

            #All hexos channels on the same spectrometer should have the same time array hence only reading in the one for Fe+22
            signalFe22spatialsumtime = np.loadtxt('/draco/u/petr/Peter_Python/'+totnameFe22+'time.csv',delimiter=',')    
            indexesbeforemaxspatialsumhexos = (np.abs(signalFe22spatialsumtime - timesliced[int(timeindexmaxvalue-indexesbeforemax)])).argmin()
            indexesaftermaxspatialsumhexos = (np.abs(signalFe22spatialsumtime - timesliced[int(timeindexmaxvalue+indexesaftermax)])).argmin()
            sigmaFe22spatialsum = np.sqrt(abs(signalFe22spatialsum[int(indexesbeforemaxspatialsumhexos):int(indexesaftermaxspatialsumhexos)])) 


            self.data = signalsliced.flatten('F')
            self.sigma = sigmasliced.flatten('F')
            self.data = np.append(self.data,signalFe22spatialsum[int(indexesbeforemaxspatialsumhexos):int(indexesaftermaxspatialsumhexos)])
            self.xhexos = signalFe22spatialsumtime[int(indexesbeforemaxspatialsumhexos):int(indexesaftermaxspatialsumhexos)]
            self.sigma = np.append(self.sigma,sigmaFe22spatialsum)

            '''
            totnameFe21='20171123.036_signalFeXXII_data_intensity_corrected_pixel_width_5'            
            signalFe21spatialsum = np.loadtxt('/draco/u/petr/Peter_Python/'+totnameFe21+'.csv',delimiter=',')
            sigmaFe21spatialsum = np.sqrt(abs(signalFe21spatialsum[int(indexesbeforemaxspatialsumhexos):int(indexesaftermaxspatialsumhexos)]))

            #self.data = signalsliced.flatten('F')
            #self.sigma = sigmasliced.flatten('F')
            self.data = np.append(self.data,signalFe21spatialsum[int(indexesbeforemaxspatialsumhexos):int(indexesaftermaxspatialsumhexos)])
            self.sigma = np.append(self.sigma,sigmaFe21spatialsum)
            '''

            '''
            totnameFe19='20171123.036_signalFeXX_data_intensity_corrected_pixel_width_5'            
            signalFe19spatialsum = np.loadtxt('/draco/u/petr/Peter_Python/'+totnameFe19+'.csv',delimiter=',')
            sigmaFe19spatialsum = np.sqrt(abs(signalFe19spatialsum[int(indexesbeforemaxspatialsumhexos):int(indexesaftermaxspatialsumhexos)]))

            self.data = np.append(self.data,signalFe19spatialsum[int(indexesbeforemaxspatialsumhexos):int(indexesaftermaxspatialsumhexos)])
            self.sigma = np.append(self.sigma,sigmaFe19spatialsum)         

            totnameFe18='20171123.036_signalFeXIX_data_intensity_corrected_pixel_width_5'            
            signalFe18spatialsum = np.loadtxt('/draco/u/petr/Peter_Python/'+totnameFe18+'.csv',delimiter=',')
            sigmaFe18spatialsum = np.sqrt(abs(signalFe18spatialsum[int(indexesbeforemaxspatialsumhexos):int(indexesaftermaxspatialsumhexos)]))            

            self.data = np.append(self.data,signalFe18spatialsum[int(indexesbeforemaxspatialsumhexos):int(indexesaftermaxspatialsumhexos)])
            self.sigma = np.append(self.sigma,sigmaFe18spatialsum)
            

            totnameFe12='20171123.036_signalFeXIII_data_intensity_corrected_pixel_width_5'            
            signalFe12spatialsum = np.loadtxt('/draco/u/petr/Peter_Python/'+totnameFe12+'.csv',delimiter=',')
            sigmaFe12spatialsum = np.sqrt(abs(signalFe12spatialsum[int(indexesbeforemaxspatialsumhexos):int(indexesaftermaxspatialsumhexos)]))            

            self.data = np.append(self.data,signalFe12spatialsum[int(indexesbeforemaxspatialsumhexos):int(indexesaftermaxspatialsumhexos)])
            self.sigma = np.append(self.sigma,sigmaFe12spatialsum)
            '''

            
            
            self.x = timesliced
            #self.xxics = signalFe24spatialsumtime[int(indexesbeforemaxspatialsumxics):int(indexesaftermaxspatialsumxics)]
            self.xrho = rhosliced

            #self.data = np.append(signalsliced.flatten('F'),signalFe24spatialsum[int(indexesbeforemaxspatialsumxics):int(indexesaftermaxspatialsumxics)]) 
            #self.data = signalsliced.flatten('F')

            #self.sigma = np.append(sigmasliced.flatten('F'),sigmaFe24spatialsum)
            #self.sigma = sigmasliced.flatten('F')

            zerovalues = (self.sigma == 0.0)
            self.sigma[zerovalues] = 1.0
            self.ionstage = 24

        
        numdloc = 4
        choiceedgerhod = 1.00
        choiceaxisrhod = 0.05

        numvloc = 4 
        choiceedgerhov = 1.00
        choiceaxisrhov = 0.05

        results = self.strahlfit(x=self.x
                                 ,numdloc=numdloc
                                 ,choiceedgerhod=choiceedgerhod
                                 ,choiceaxisrhod=choiceaxisrhod
                                 ,numvloc=numvloc
                                 ,choiceedgerhov=choiceedgerhov
                                 ,choiceaxisrhov=choiceaxisrhov)

        self.print_results(results)

        print('timeindexmaxvalue: ', *timeindexmaxvalue, '   time[timeindexmaxvalue]: ', *time[timeindexmaxvalue])
        print('indexesbeforemax: ', indexesbeforemax, '   first time @ indexesbeforemax: ', time[timeindexmaxvalue-indexesbeforemax])
        print('indexesaftermax: ', indexesaftermax, '   last time @ indexesaftermax: ', time[int(timeindexmaxvalue+indexesaftermax)])#time[timeindexmaxvalue-indexesaftermax])

        self.plot_results(results)

        transporttimeresults=self.runanalysis(name=results['netcdffilename'],ionstage=24,signalflag=True)     

        residualdataname = name + '_wyzemissivity_residual'+ '_d' + str(results['coeff'][0]) + '_v'+ str(results['coeff'][numdloc+1]) +'_TSV3'
        print(residualdataname)
        
        rhovaluescombined = np.append(results['rhodval'],results['rhovval'])   
        rhovaluescombinedwithzeros = np.append(rhovaluescombined,np.asarray([0.0]*(len(results['coeff'])-len(results['rhodval'])-len(results['rhovval']))))
       

        np.savetxt('/draco/u/petr/'+name+'_wyzemissivity_data'+ '_d' + str(results['coeff'][0]) + '_v'+ str(results['coeff'][numdloc+1])+'_TSV3_spatialnum_'+str(numdloc)+'_axis_'+str(choiceaxisrhod)+'_edge_'+str(choiceedgerhod)+'.csv',self.data,delimiter=',')
        np.savetxt('/draco/u/petr/'+name+'_wyzemissivity_fit'+ '_d' + str(results['coeff'][0]) + '_v'+ str(results['coeff'][numdloc+1])+'_TSV3_spatialnum_'+str(numdloc)+'_axis_'+str(choiceaxisrhod)+'_edge_'+str(choiceedgerhod)+'.csv',results['fit'],delimiter=',')
        np.savetxt('/draco/u/petr/'+name+'_wyzemissivity_sigma'+ '_d' + str(results['coeff'][0]) + '_v'+ str(results['coeff'][numdloc+1])+'_TSV3_spatialnum_'+str(numdloc)+'_axis_'+str(choiceaxisrhod)+'_edge_'+str(choiceedgerhod)+'.csv',self.sigma,delimiter=',')
        np.savetxt('/draco/u/petr/'+name+'_wyzemissivity_fitparams'+ '_d' + str(results['coeff'][0]) + '_v'+ str(results['coeff'][numdloc+1])+'_TSV3_spatialnum_'+str(numdloc)+'_axis_'+str(choiceaxisrhod)+'_edge_'+str(choiceedgerhod)+'.csv',[rhovaluescombinedwithzeros,results['coeff'],results['params'],results['perror'],results['minparamstepsize'],[results['summedchisquared']]*len(results['coeff'])],delimiter=',')
        np.savetxt('/draco/u/petr/'+residualdataname+'time_spatialnum_'+str(numdloc)+'_axis_'+str(choiceaxisrhod)+'_edge_'+str(choiceedgerhod)+'.csv',timesliced,delimiter=',')
        np.savetxt('/draco/u/petr/'+residualdataname+'_spatialnum_'+str(numdloc)+'_axis_'+str(choiceaxisrhod)+'_edge_'+str(choiceedgerhod)+'.csv',self.data-results['fit'],delimiter=',')
        print('All done')

        return results
