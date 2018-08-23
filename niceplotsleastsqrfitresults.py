# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 18:57:16 2018

@author: petr
"""
import numpy as np
import raw_data_access
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
import os

def grabsubfolderandfilenames(directory=None):
    roots = []
    dirs = []
    files = []
    
    for root, direct, file in os.walk(directory):
        roots.append(roots)
        dirs.append(direct)
        files.append(file)
        
    return {'roots': roots
            ,'dirs': dirs
            ,'files': files}

def loadoutputdata(outputdirectory=None):
    names = grabsubfolderandfilenames(outputdirectory)
    folders = []
    strahldata = []
    #length = len(names['dirs'][0])
    counter = 0
    for name in names['dirs'][0]:
        folders.append(name)
        if counter == 0:
            fit = np.loadtxt(outputdirectory+'/'+name+'/'+'fit.csv',delimiter=',')
            residual = np.loadtxt(outputdirectory+'/'+name+'/'+'residual.csv',delimiter=',')
            fitparams = np.loadtxt(outputdirectory+'/'+name+'/'+'fitparams.csv',delimiter=',')
            
            data = np.loadtxt(outputdirectory+'/'+name+'/'+'data.csv',delimiter=',')
            sigma = np.loadtxt(outputdirectory+'/'+name+'/'+'sigma.csv',delimiter=',')
            time = np.loadtxt(outputdirectory+'/'+name+'/'+'time.csv',delimiter=',')
            strahldata.append(raw_data_access.readstrahlnetcdf(outputdirectory+'/'+name+'/'+names['files'][counter+1][0]))
            counter=counter + 1
            print('name is: ',name)
            print('fit.shape is: ', fit.shape)
            print('fitparams.shape is: ', fitparams.shape)
        else:
            fit = np.vstack([fit,np.loadtxt(outputdirectory+'/'+name+'/'+'fit.csv',delimiter=',')])
            residual = np.vstack([residual,np.loadtxt(outputdirectory+'/'+name+'/'+'residual.csv',delimiter=',')])
            fitparams = np.vstack([fitparams,np.loadtxt(outputdirectory+'/'+name+'/'+'fitparams.csv',delimiter=',')])

            
            data = np.vstack([data,np.loadtxt(outputdirectory+'/'+name+'/'+'data.csv',delimiter=',')])
            sigma = np.vstack([sigma,np.loadtxt(outputdirectory+'/'+name+'/'+'sigma.csv',delimiter=',')])
            time = np.vstack([time,np.loadtxt(outputdirectory+'/'+name+'/'+'time.csv',delimiter=',')])
            strahldata.append(raw_data_access.readstrahlnetcdf(outputdirectory+'/'+name+'/'+names['files'][counter+1][0]))
            counter = counter+1
            print('name is: ',name)
            print('fit.shape is: ', fit.shape)
            print('fitparams.shape is: ', fitparams.shape)
    
    
    return {'fit':fit
            ,'residual':residual
            ,'fitparams':fitparams
            ,'data':data
            ,'sigma':sigma
            ,'time':time
            ,'strahldata':strahldata
            ,'names':folders}
    
def loadinputdata(inputdirectory
                  ,sigtype):
    
    #inputdirectory = 'C:/Users/petr/Desktop/Peter_Python/'
    if (sigtype != 'Ar17' and sigtype != 'qsx'):
        totnameFe22='20171123.036_signalFeXXIII_data_intensity_corrected_pixel_width_5'
        signalFe22spatialsumtime = np.loadtxt(inputdirectory+totnameFe22+'time.csv',delimiter=',')
        #indexesbeforemaxspatialsumhexos = (np.abs(signalFe22spatialsumtime - timesliced[int(timeindexmaxvalue-indexesbeforemax)])).argmin()
        #indexesaftermaxspatialsumhexos = (np.abs(signalFe22spatialsumtime - timesliced[int(timeindexmaxvalue+indexesaftermax)])).argmin()
        alltimesig = signalFe22spatialsumtime
    if sigtype == 'Ar17':
        totnameFe24='171123036ar17data'            
        signalFe24spatialsum = np.loadtxt(inputdirectory+totnameFe24+'.csv',delimiter=',')
        signalFe24spatialsumtime = np.loadtxt(inputdirectory+totnameFe24+'time.csv',delimiter=',')
        
        backgroundsigmaFe24 = 0.178614366348
        scalingfactorFe24 = 2.70237130428
        sigmaFe24spatialsum = np.sqrt(backgroundsigmaFe24**2.0 + (scalingfactorFe24**2.0)*abs(signalFe24spatialsum)) 
        
        alltimesig = signalFe24spatialsumtime
        alldatasig = signalFe24spatialsum
        alldatasigma = sigmaFe24spatialsum
    if sigtype == 'FeXIII':
        totnameFe12='20171123.036_signalFeXIII_data_intensity_corrected_pixel_width_5'
        signalFe12spatialsum = np.loadtxt(inputdirectory+totnameFe12+'.csv',delimiter=',')
        
        backgroundsigmaFe12 = 155.043722909
        scalingfactorFe12 = 0.961643528038
        sigmaFe12spatialsum = np.sqrt(backgroundsigmaFe12**2.0 + (scalingfactorFe12**2.0)*abs(signalFe12spatialsum))
                    
        alldatasig = signalFe12spatialsum
        alldatasigma = sigmaFe12spatialsum
    if sigtype == 'FeXIX':
        totnameFe18='20171123.036_signalFeXIX_data_intensity_corrected_pixel_width_5'
        signalFe18spatialsum = np.loadtxt(inputdirectory+totnameFe18+'.csv',delimiter=',')
        
        backgroundsigmaFe18 = 151.809466853
        scalingfactorFe18 = 2.35933640913
        sigmaFe18spatialsum = np.sqrt(backgroundsigmaFe18**2.0 + (scalingfactorFe18**2.0)*abs(signalFe18spatialsum))            
                    
        
        alldatasig = signalFe18spatialsum
        alldatasigma = sigmaFe18spatialsum
    if sigtype == 'FeXX':
        totnameFe19='20171123.036_signalFeXX_data_intensity_corrected_pixel_width_5'
        signalFe19spatialsum = np.loadtxt(inputdirectory+totnameFe19+'.csv',delimiter=',')
                
        backgroundsigmaFe19 = 154.44283169
        scalingfactorFe19 = 3.52902275406
        sigmaFe19spatialsum = np.sqrt(backgroundsigmaFe19**2.0 + (scalingfactorFe19**2.0)*abs(signalFe19spatialsum))            
                    
        
        alldatasig = signalFe19spatialsum
        alldatasigma = sigmaFe19spatialsum
    if sigtype == 'FeXXII':
        totnameFe21='20171123.036_signalFeXXII_data_intensity_corrected_pixel_width_5'
        signalFe21spatialsum = np.loadtxt(inputdirectory+totnameFe21+'.csv',delimiter=',')
        
                        
        backgroundsigmaFe21 = 154.463064814
        scalingfactorFe21 = 5.5269293208
        sigmaFe21spatialsum = np.sqrt(backgroundsigmaFe21**2.0 + (scalingfactorFe21**2.0)*abs(signalFe21spatialsum))            
                    
        
        alldatasig = signalFe21spatialsum
        alldatasigma = sigmaFe21spatialsum
    
    return {'alldatasig':alldatasig
            ,'alltimesig':alltimesig
            ,'alldatasigma':alldatasigma}

def weightedresidual_plot(alltimesig=None
                          ,alldatasig=None
                          ,alldatasigma=None
                          ,fit=None
                          ,sigtype=None
                          ,timeindexes=None
                          ,timeindexflag=False
                          ,fsize=None
                          ,title=None):
    
    #alltimesig = fitresults['time']
    #alldatasig = fitresults['data']
    #alldatasigma = fitresults['sigma']
    
    '''
    if sigtype == 'Ar17':
        yerrfit = self.linearandexponentialmodel(coeff=fitresults['perror'],x=alltimesig)
    else:
        yerrfit = self.linearmodel(coeff=results['perror'],x=alltimesig)
    '''
    
    fig = plt.figure(figsize=fsize)
    gs = gridspec.GridSpec(2,1,height_ratios=[2,1])
    ax1 = plt.subplot(gs[0])
    fig.suptitle(title,fontsize=15)
    ax1.errorbar(alltimesig,alldatasig,yerr=alldatasigma, fmt='bo', ms=2,zorder=0)
    #ax1.errorbar(alltimesig, fitresults['fit'],yerr = yerrfit, color='red',zorder=1)
    ax1.scatter(alltimesig, fit, color='red',s=2,zorder=2)
    if timeindexflag == True:
        for index in timeindexes:
            ax1.axvline(x=alltimesig[index],color='k', linestyle='--')
    ax1.set_ylabel('Intensity (arb)',fontsize=15)
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2 = plt.subplot(gs[1],sharex=ax1)
    ax2.scatter(alltimesig,(1/alldatasigma)*(alldatasig-fit))
    ax2.set_ylabel('Weighted residual',fontsize=15)
    ax2.set_xlabel('Time (s)',fontsize=15)
    plt.subplots_adjust(hspace=.0)
    
def densityfractionalabundanceplot(rho=None
                                   ,chargestates=None
                                   ,density=None
                                   ,timeindexes=None
                                   ,strahltime=None
                                   ,sigtype=None
                                   ,directory=None
                                   ,fsize=None
                                   ,title=None):
    
    legendbasestr = '{}'
    #colorlist = ['b','g','r','c','m']
    
    for jj in timeindexes:
        plt.figure(figsize=fsize)
        ax = plt.gca()
        for ii in chargestates:
            color = next(ax._get_lines.prop_cycler)['color']
            plt.plot(rho,abs(density[jj,ii]/density[jj].max()),color=color,marker='o',label=legendbasestr.format('+ '+str(ii)))
            plt.axvline(x=rho[np.argmax(density[jj,ii])],color=color, linestyle='--')
            plt.title(title+'\n'+'Time: '+str(strahltime[jj])+' s',fontsize=15)
            print('title is: '+title+'\n'+'Time: '+str(strahltime[jj])+' s '+'chargestate: + '+str(ii))
            print('Density max is '+ str(density[jj].max()))
            print('Density min is '+ str(density[jj].min()))
            print('Density min/max is '+ str(density[jj].min()/density[jj].max()))
        plt.legend()
        #plt.ylim(0,np.max(diff)*5)
        plt.yscale('log')
        plt.xlabel('Rho (r/a)',fontsize=15)
        plt.ylabel('Normalized density (arb)',fontsize=15)


def radiationfractionalabundanceplot(rho=None
                                     ,chargestates=None
                                     ,radiation=None
                                     ,timeindexes=None
                                     ,strahltime=None
                                     ,sigtype=None
                                     ,directory=None
                                     ,fsize=None
                                     ,title=None):
    
    legendbasestr = '{}'
    #colorlist = ['b','g','r','c','m']
    
    chargestatesindexinatomfile = [0,1,2,3,5]
    counter = 0
    for jj in timeindexes:
        plt.figure(figsize=fsize)
        ax = plt.gca()
        for ii in chargestatesindexinatomfile:
            color = next(ax._get_lines.prop_cycler)['color']
            plt.plot(rho,abs(radiation[jj,ii]/radiation[jj].max()),color=color,marker='o',label=legendbasestr.format('+ '+str(chargestates[counter])))
            #plt.plot(rho,radiation[jj,ii]/radiation[jj].max(),color=color,marker='o',label=legendbasestr.format('+ '+str(chargestates[counter])))
            plt.axvline(x=rho[np.argmax(radiation[jj,ii])],color=color, linestyle='--')
            plt.title(title+'\n'+'Time: '+str(strahltime[jj])+' s',fontsize=15)
            print('title is: '+title+'\n'+'Time: '+str(strahltime[jj])+' s '+'chargestate: + '+str(ii))
            print('Radiation max is '+ str(radiation[jj].max()))
            print('Radiation min is '+ str(radiation[jj].min()))
            print('Radiation min/max is '+ str(radiation[jj].min()/radiation[jj].max()))
            counter = counter + 1
        plt.legend()
        #plt.ylim(0,np.max(diff)*5)
        plt.yscale('log')
        plt.xlabel('Rho (r/a)',fontsize=15)
        plt.ylabel('Normalized radiation (arb)',fontsize=15)
        counter = 0


def plotalldata(directory=None
                ,sigtype=None):
    
    fsize=(10.0,9.0)
    #linewidth = 80
    #titlesize = 15
    #labelsize = 15
    #ticksize = 15
    numdloc = 3
    numvloc = 3
    numberofchargestates = 5 
    numberoftimepoints = 4
    chargestatesofinterest = np.zeros(numberofchargestates)
    timesofinterest = np.zeros(numberoftimepoints)
    timeindexes = np.zeros(numberoftimepoints)
    timeindexesoriginal = np.zeros(numberoftimepoints)
    
    chargestatesofinterest[0] = 12
    chargestatesofinterest[1] = 18
    chargestatesofinterest[2] = 19
    chargestatesofinterest[3] = 21
    chargestatesofinterest[4] = 24
    
    timesofinterest[0] = 2.36
    timesofinterest[1] = 2.40
    timesofinterest[2] = 2.44
    timesofinterest[3] = 2.48
    
    resultdata = loadoutputdata(directory)    
    
    '''
    keywords = []
    
    for keyword in resultdata:
        keywords.append(keyword)
    '''
    if sigtype == 'invert':
        print('')
    if sigtype == 'Ar17' or sigtype == 'qsx':
        print('')
    else:
        totnameFe22='20171123.036_signalFeXXIII_data_intensity_corrected_pixel_width_5'
        signalFe22spatialsumtime = np.loadtxt('C:/Users/petr/Desktop/Peter_Python/'+totnameFe22+'time.csv',delimiter=',')
        indexesbeforemaxspatialsumhexos = (np.abs(signalFe22spatialsumtime - resultdata['time'][0][0])).argmin()
        indexesaftermaxspatialsumhexos = (np.abs(signalFe22spatialsumtime - resultdata['time'][0][-1])).argmin()
        time = signalFe22spatialsumtime[int(indexesbeforemaxspatialsumhexos):int(indexesaftermaxspatialsumhexos)]
        #print('time.shape is: ',time.shape)
        for ii in range(numberoftimepoints):
            timeindexes[ii] = np.abs(time - timesofinterest[ii]).argmin()
            timeindexesoriginal[ii] = np.abs(resultdata['strahldata'][0][1]['time'] - timesofinterest[ii]).argmin() #careful if timing changes between strahl files this is not accurate

    numberoffolders = len(resultdata['names'])
    for ii in range(numberoffolders):  
        title = directory+'\n'+resultdata['names'][ii]+'\n'+'Total Chi squared: '+str(resultdata['fitparams'][5+(6*ii)][0])
        weightedresidual_plot(alltimesig=time
                              ,alldatasig=resultdata['data'][ii]
                              ,alldatasigma=resultdata['sigma'][ii]
                              ,fit=resultdata['fit'][ii]
                              ,timeindexes=timeindexes
                              ,timeindexflag=True
                              ,sigtype=sigtype
                              ,fsize=fsize
                              ,title=title)
        '''
        densityfractionalabundanceplot(rho=resultdata['strahldata'][ii][1]['rho_poloidal_grid']
                                       ,chargestates=chargestatesofinterest
                                       ,density=resultdata['strahldata'][ii][1]['impurity_density']
                                       ,timeindexes=timeindexesoriginal
                                       ,strahltime=resultdata['strahldata'][ii][1]['time']
                                       ,sigtype=sigtype
                                       ,directory=directory
                                       ,fsize=fsize
                                       ,title=title)
        '''
        '''
        radiationfractionalabundanceplot(rho=resultdata['strahldata'][ii][1]['rho_poloidal_grid']
                                         ,chargestates=chargestatesofinterest
                                         ,radiation=resultdata['strahldata'][ii][1]['diag_lines_radiation']
                                         ,timeindexes=timeindexesoriginal
                                         ,strahltime=resultdata['strahldata'][ii][1]['time']
                                         ,sigtype=sigtype
                                         ,directory=directory
                                         ,fsize=fsize
                                         ,title=title)
        '''
        
    legendbasestr = '{}    {}'
    #colorlist = ['b','g','r','c','m']
    
    plt.figure(figsize=fsize)
    ax = plt.gca()
    for ii in range(numberoffolders):
        color = next(ax._get_lines.prop_cycler)['color']
        radiation = resultdata['strahldata'][ii][1]['diag_lines_radiation']
        diff = (10.0**-4.0)*resultdata['strahldata'][ii][1]['anomal_diffusion']
        rho = resultdata['strahldata'][ii][1]['rho_poloidal_grid']
        #plt.errorbar(rho, diff[0], fmt='bo', ms=8,label=legendbasestr.format(str(resultdata['names'][ii])))
        plt.errorbar(rho, diff[0],color=color,marker='o', ms=8,label=legendbasestr.format(str(resultdata['fitparams'][5+(6*ii)][0]),str(resultdata['names'][ii])))
        plt.errorbar(resultdata['fitparams'][0+(6*ii)][:numdloc],resultdata['fitparams'][2+(6*ii)][:numdloc],yerr=resultdata['fitparams'][3+(6*ii)][:numdloc], fmt='ko', ms=7, ecolor=color)
        for jj in timeindexesoriginal:
            plt.axvline(x=rho[np.argmax(radiation[jj,3])],color=color, linestyle='--')
        plt.fill_between(resultdata['fitparams'][0+(6*ii)][:numdloc]
                         ,(resultdata['fitparams'][2+(6*ii)][:numdloc]+resultdata['fitparams'][3+(6*ii)][:numdloc])
                         ,(resultdata['fitparams'][2+(6*ii)][:numdloc]-resultdata['fitparams'][3+(6*ii)][:numdloc])
                         ,color=color
                         ,alpha=0.3)
    plt.legend()
    #plt.ylim(0,np.max(diff)*5)
    #plt.xlim(0.6,0.9)
    plt.xlabel('Rho (r/a)',fontsize=15)
    plt.ylabel('Diffusion coefficient (m^2/s)',fontsize=15)
    plt.title(directory,fontsize=15)

    plt.figure(figsize=fsize)
    ax = plt.gca()
    for ii in range(numberoffolders):
        color = next(ax._get_lines.prop_cycler)['color']
        radiation = resultdata['strahldata'][ii][1]['diag_lines_radiation']
        conv = (10.0**-2.0)*resultdata['strahldata'][ii][1]['anomal_drift']
        rho = resultdata['strahldata'][ii][1]['rho_poloidal_grid']
        #plt.errorbar(rho, diff[0], fmt='bo', ms=8,label=legendbasestr.format(str(resultdata['names'][ii])))
        plt.errorbar(rho, conv[0],color=color,marker='o', ms=8,label=legendbasestr.format(str(resultdata['fitparams'][5+(6*ii)][0]),str(resultdata['names'][ii])))
        plt.errorbar(resultdata['fitparams'][0+(6*ii)][numdloc:numdloc+numvloc],resultdata['fitparams'][2+(6*ii)][numdloc:numdloc+numvloc],yerr=resultdata['fitparams'][3+(6*ii)][numdloc:numdloc+numvloc], fmt='ko', ms=7, ecolor=color)
        for jj in timeindexesoriginal:
            plt.axvline(x=rho[np.argmax(radiation[jj,3])],color=color, linestyle='--')
        plt.fill_between(resultdata['fitparams'][0+(6*ii)][numdloc:numdloc+numvloc]
                 ,(resultdata['fitparams'][2+(6*ii)][numdloc:numdloc+numvloc]+resultdata['fitparams'][3+(6*ii)][numdloc:numdloc+numvloc])
                 ,(resultdata['fitparams'][2+(6*ii)][numdloc:numdloc+numvloc]-resultdata['fitparams'][3+(6*ii)][numdloc:numdloc+numvloc])
                 ,color=color
                 ,alpha=0.3)
    plt.legend()
    #plt.ylim(0,np.max(diff)*5)
    #plt.xlim(0.6,0.9)
    plt.xlabel('Rho (r/a)',fontsize=15)
    plt.ylabel('Convective velocity (m/s)',fontsize=15)
    plt.title(directory,fontsize=15)
    
    
    size = 200
    
    plt.figure(figsize=fsize)
    ax = plt.gca()
    for ii in range(numberoffolders):
        color = next(ax._get_lines.prop_cycler)['color']
        name = directory+'/'+str(resultdata['names'][ii])+'/op12a_171123036_FeLBO_chisqd.dat'
        if os.path.exists(name):
            chisqrddata = np.loadtxt(name)
            plt.plot(chisqrddata[:,0],chisqrddata[:,-1],marker='o',color=color,label=legendbasestr.format(str(resultdata['fitparams'][5+(6*ii)][0]),str(resultdata['names'][ii])))
            plt.scatter(chisqrddata[0,0],chisqrddata[0,-1],s=size,marker='v',color=color)
            plt.scatter(chisqrddata[-1,0],chisqrddata[-1,-1],s=size,marker='^',color=color)
    plt.legend()
    #plt.ylim(0,np.max(diff)*5)
    #plt.xlim(0.6,0.9)
    plt.xlabel('Axis Diffusion paramter (m^2/s)',fontsize=15)
    plt.ylabel('Total chisquared',fontsize=15)
    plt.title(directory,fontsize=15)

    plt.figure(figsize=fsize)
    ax = plt.gca()
    for ii in range(numberoffolders):
        color = next(ax._get_lines.prop_cycler)['color']
        name = directory+'/'+str(resultdata['names'][ii])+'/op12a_171123036_FeLBO_chisqd.dat'
        if os.path.exists(name):
            chisqrddata = np.loadtxt(name)
            plt.plot(chisqrddata[:,numdloc-1],chisqrddata[:,-1],marker='o',color=color,label=legendbasestr.format(str(resultdata['fitparams'][5+(6*ii)][0]),str(resultdata['names'][ii])))
            plt.scatter(chisqrddata[0,numdloc-1],chisqrddata[0,-1],s=size,marker='v',color=color)
            plt.scatter(chisqrddata[-1,numdloc-1],chisqrddata[-1,-1],s=size,marker='^',color=color)
    plt.legend()
    #plt.ylim(0,np.max(diff)*5)
    #plt.xlim(0.6,0.9)
    plt.xlabel('Edge Diffusion paramter (m^2/s)',fontsize=15)
    plt.ylabel('Total chisquared',fontsize=15)
    plt.title(directory,fontsize=15)

    plt.figure(figsize=fsize)
    ax = plt.gca()
    for ii in range(numberoffolders):
        color = next(ax._get_lines.prop_cycler)['color']
        name = directory+'/'+str(resultdata['names'][ii])+'/op12a_171123036_FeLBO_chisqd.dat'
        if os.path.exists(name):
            chisqrddata = np.loadtxt(name)
            plt.plot(chisqrddata[:,numdloc+numvloc-1],chisqrddata[:,-1],marker='o',color=color,label=legendbasestr.format(str(resultdata['fitparams'][5+(6*ii)][0]),str(resultdata['names'][ii])))
            plt.scatter(chisqrddata[0,numdloc+numvloc-1],chisqrddata[0,-1],s=size,marker='v',color=color)
            plt.scatter(chisqrddata[-1,numdloc+numvloc-1],chisqrddata[-1,-1],s=size,marker='^',color=color)
    plt.legend()
    #plt.ylim(0,np.max(diff)*5)
    #plt.xlim(0.6,0.9)
    plt.xlabel('Edge Convection paramter (m/s)',fontsize=15)
    plt.ylabel('Total chisquared',fontsize=15)
    plt.title(directory,fontsize=15)

    plt.figure(figsize=fsize)
    ax = plt.gca()
    for ii in range(numberoffolders):
        color = next(ax._get_lines.prop_cycler)['color']
        name = directory+'/'+str(resultdata['names'][ii])+'/op12a_171123036_FeLBO_chisqd.dat'
        if os.path.exists(name):
            chisqrddata = np.loadtxt(name)
            plt.plot(chisqrddata[:,numdloc+numvloc+5],chisqrddata[:,-1],marker='o',color=color,label=legendbasestr.format(str(resultdata['fitparams'][5+(6*ii)][0]),str(resultdata['names'][ii])))
            plt.scatter(chisqrddata[0,numdloc+numvloc+5],chisqrddata[0,-1],s=size,marker='v',color=color)
            plt.scatter(chisqrddata[-1,numdloc+numvloc+5],chisqrddata[-1,-1],s=size,marker='^',color=color)
    plt.legend()
    #plt.ylim(0,np.max(diff)*5)
    #plt.xlim(0.6,0.9)
    plt.xlabel('Scale radiation signal parameter',fontsize=15)
    plt.ylabel('Total chisquared',fontsize=15)
    plt.title(directory,fontsize=15)
    
    
    legendbasestr2 = '{}  {}  {}'
    
    plt.figure(figsize=fsize)
    ax = plt.gca()
    for ii in range(numberoffolders):
        color = next(ax._get_lines.prop_cycler)['color']
        name = directory+'/'+str(resultdata['names'][ii])+'/op12a_171123036_FeLBO_chisqd.dat'
        if os.path.exists(name):
            chisqrddata = np.loadtxt(name)
            plt.plot(chisqrddata[:,numdloc+numvloc+6],chisqrddata[:,-1],marker='o',color=color,label=legendbasestr2.format(str(resultdata['fitparams'][5+(6*ii)][0]),str(resultdata['fitparams'][2+(6*ii)][numdloc+numvloc+6]),str(resultdata['names'][ii])))
            plt.scatter(chisqrddata[0,numdloc+numvloc+6],chisqrddata[0,-1],s=size,marker='v',color=color)
            plt.scatter(chisqrddata[-1,numdloc+numvloc+6],chisqrddata[-1,-1],s=size,marker='^',color=color)
    plt.legend()
    #plt.ylim(0,np.max(diff)*5)
    #plt.xlim(0.6,0.9)
    plt.xlabel('1st LBO pulse timing parameter',fontsize=15)
    plt.ylabel('Total chisquared',fontsize=15)
    plt.title(directory,fontsize=15)
    
    plt.figure(figsize=fsize)
    ax = plt.gca()
    for ii in range(numberoffolders):
        color = next(ax._get_lines.prop_cycler)['color']
        name = directory+'/'+str(resultdata['names'][ii])+'/op12a_171123036_FeLBO_chisqd.dat'
        if os.path.exists(name):
            chisqrddata = np.loadtxt(name)
            plt.plot(chisqrddata[:,numdloc+numvloc+7],chisqrddata[:,-1],marker='o',color=color,label=legendbasestr2.format(str(resultdata['fitparams'][5+(6*ii)][0]),str(resultdata['fitparams'][2+(6*ii)][numdloc+numvloc+7]),str(resultdata['names'][ii])))
            plt.scatter(chisqrddata[0,numdloc+numvloc+7],chisqrddata[0,-1],s=size,marker='v',color=color)
            plt.scatter(chisqrddata[-1,numdloc+numvloc+7],chisqrddata[-1,-1],s=size,marker='^',color=color)
    plt.legend()
    #plt.ylim(0,np.max(diff)*5)
    #plt.xlim(0.6,0.9)
    plt.xlabel('2nd LBO pulse timing parameter',fontsize=15)
    plt.ylabel('Total chisquared',fontsize=15)
    plt.title(directory,fontsize=15)






#######Loading in raw data
#nameinvert='w7x_ar17_171123036_xics_temp_inverted_WEmiss'
#signalinvert=np.loadtxt('C:/Users/petr/Desktop/Peter_Python/'+nameinvert+'.csv',delimiter=',')
#timeinvert=10**(-3.0)*np.loadtxt('C:/Users/petr/Desktop/Peter_Python/'+nameinvert+'time.csv',delimiter=',')
#rhoinvert=np.loadtxt('C:/Users/petr/Desktop/Peter_Python/'+nameinvert+'rho.csv',delimiter=',')
#sigmainvert=np.loadtxt('C:/Users/petr/Desktop/Peter_Python/'+nameinvert+'sigma.csv',delimiter=',')
#
#timeindexmaxvalue = (np.where(signalinvert/signalinvert.max()==(signalinvert/signalinvert.max()).max()))[0]
#indexesbeforemax = timeindexmaxvalue#50
#indexesaftermax = -timeindexmaxvalue-1#100
#
#startspatialindex = 0
#endspatialindex = -1
#signalsliced = signalinvert[int(timeindexmaxvalue-indexesbeforemax):int(timeindexmaxvalue+indexesaftermax),startspatialindex:endspatialindex] #/ (signal[int(timeindexmaxvalue-indexesbeforemax):int(timeindexmaxvalue+indexesaftermax),startspatialindex:endspatialindex]).max()
#sigmasliced = sigmainvert[int(timeindexmaxvalue-indexesbeforemax):int(timeindexmaxvalue+indexesaftermax),startspatialindex:endspatialindex]
#
#timesliced = timeinvert[int(timeindexmaxvalue-indexesbeforemax):int(timeindexmaxvalue+indexesaftermax)]
#rhosliced = rhoinvert[startspatialindex:endspatialindex]
#
#lengthbeforetile = len(timesliced)
#timesliced = np.tile(timesliced,len(rhosliced))
#lengthbeforesumsignal = len(timesliced)
#signalsliced = signalsliced.flatten('F')
#sigmasliced = sigmasliced.flatten('F')
#
#
#totnameFe22='20171123.036_signalFeXXIII_data_intensity_corrected_pixel_width_5'
#signalFe22spatialsumtime = np.loadtxt('C:/Users/petr/Desktop/Peter_Python/'+totnameFe22+'time.csv',delimiter=',')
#indexesbeforemaxspatialsumhexos = (np.abs(signalFe22spatialsumtime - timesliced[int(timeindexmaxvalue-indexesbeforemax)])).argmin()
#indexesaftermaxspatialsumhexos = (np.abs(signalFe22spatialsumtime - timesliced[int(timeindexmaxvalue+indexesaftermax)])).argmin()
#
#
#totnameFe21='20171123.036_signalFeXXII_data_intensity_corrected_pixel_width_5'
#signalFe21spatialsum = np.loadtxt('C:/Users/petr/Desktop/Peter_Python/'+totnameFe21+'.csv',delimiter=',')
##sigmaFe21spatialsum = np.sqrt(abs(signalFe21spatialsum[int(indexesbeforemaxspatialsumhexos):int(indexesaftermaxspatialsumhexos)]))
#
#totnameFe19='20171123.036_signalFeXX_data_intensity_corrected_pixel_width_5'
#signalFe19spatialsum = np.loadtxt('C:/Users/petr/Desktop/Peter_Python/'+totnameFe19+'.csv',delimiter=',')
##sigmaFe19spatialsum = np.sqrt(abs(signalFe19spatialsum[int(indexesbeforemaxspatialsumhexos):int(indexesaftermaxspatialsumhexos)]))
#
#totnameFe18='20171123.036_signalFeXIX_data_intensity_corrected_pixel_width_5'
#signalFe18spatialsum = np.loadtxt('C:/Users/petr/Desktop/Peter_Python/'+totnameFe18+'.csv',delimiter=',')
##sigmaFe18spatialsum = np.sqrt(abs(signalFe18spatialsum[int(indexesbeforemaxspatialsumhexos):int(indexesaftermaxspatialsumhexos)]))
#
#tottimehexos = signalFe22spatialsumtime[int(indexesbeforemaxspatialsumhexos):int(indexesaftermaxspatialsumhexos)]
#
#######Loading in fit result data
##path = 'C:/Users/petr/Desktop/New folder/Hexos21_19_18_fixedbackground_no_summed_xics/edge_connection_lengths_changed/1st_order_spline_12_points_axis_and_edge_points_included'
#path = 'C:/Users/petr/Desktop/New folder/Wemissinvertonly/1st_order_spline_12_points_axis_and_edge_points_included'
#
#name='/d0.00175335266285126_v0.04261235473357545'
#dimension, data, attribute = raw_data_access.readstrahlnetcdf(path+name)
#name2='_2/d0.007468402908629913_v0.1952275719096626'  # 
#dimension2, data2, attribute2 = raw_data_access.readstrahlnetcdf(path+name2)
#'''
#name3='_3/d0.24221405040999366_v0.5554123996005423' # 
#dimension3, data3, attribute3 = raw_data_access.readstrahlnetcdf(path+name3)
#name4='_4/d2.6623617462728664_v-0.3490281788694198' # 
#dimension4, data4, attribute4 = raw_data_access.readstrahlnetcdf(path+name4)
#name5='_5/d0.8149738447523497_v1.9855827550701202' # 
#dimension5, data5, attribute5 = raw_data_access.readstrahlnetcdf(path+name5)
#'''
#
#resultsname=np.loadtxt(path+'/fit.csv',delimiter=',')
#resultsnamedata=np.loadtxt(path+'/data.csv',delimiter=',')
#resultsnamesigma=np.loadtxt(path+'/sigma.csv',delimiter=',')
#resultsnametime=np.loadtxt(path+'/time.csv',delimiter=',')
#fitparams_name=np.loadtxt(path+'/fitparams.csv',delimiter=',')
#
#resultsname2=np.loadtxt(path+'_2/fit.csv',delimiter=',')
#resultsnamedata2=np.loadtxt(path+'_2/data.csv',delimiter=',')
#resultsnamesigma2=np.loadtxt(path+'_2/sigma.csv',delimiter=',')
#resultsnametime2=np.loadtxt(path+'_2/time.csv',delimiter=',')
#fitparams_name2=np.loadtxt(path+'_2/fitparams.csv',delimiter=',')
#
#'''
#resultsname3=np.loadtxt(path+'_3/fit.csv',delimiter=',')
#resultsnamedata3=np.loadtxt(path+'_3/data.csv',delimiter=',')
#resultsnamesigma3=np.loadtxt(path+'_3/sigma.csv',delimiter=',')
#resultsnametime3=np.loadtxt(path+'_3/time.csv',delimiter=',')
#fitparams_name3=np.loadtxt(path+'_3/fitparams.csv',delimiter=',')
#
#resultsname4=np.loadtxt(path+'_4/fit.csv',delimiter=',')
#resultsnamedata4=np.loadtxt(path+'_4/data.csv',delimiter=',')
#resultsnamesigma4=np.loadtxt(path+'_4/sigma.csv',delimiter=',')
#resultsnametime4=np.loadtxt(path+'_4/time.csv',delimiter=',')
#fitparams_name4=np.loadtxt(path+'_4/fitparams.csv',delimiter=',')
#
#resultsname5=np.loadtxt(path+'_5/fit.csv',delimiter=',')
#resultsnamedata5=np.loadtxt(path+'_5/data.csv',delimiter=',')
#resultsnamesigma5=np.loadtxt(path+'_5/sigma.csv',delimiter=',')
#resultsnametime5=np.loadtxt(path+'_5/time.csv',delimiter=',')
#fitparams_name5=np.loadtxt(path+'_5/fitparams.csv',delimiter=',')
#'''
#
#
######Start of the manipulation of result data
#timeslice = 30
#numofradlines = len(data['diag_lines_radiation'][0,:,0])
#if numofradlines < 3:
#    signalradFe24 = data['diag_lines_radiation'][:,1]
#else:
#    #signalradFe12 = data['diag_lines_radiation'][:,0]
#    signalradFe18 = data['diag_lines_radiation'][:,1]
#    signalradFe19 = data['diag_lines_radiation'][:,2]
#    signalradFe21 = data['diag_lines_radiation'][:,3]
#    #signalradFe22 = data['diag_lines_radiation'][:,4]
#    signalradFe24 = data['diag_lines_radiation'][:,5]
#    #signalarraysumspatialradFe12 = np.sum(signalradFe12,axis=1)
#    signalarraysumspatialradFe18 = np.sum(signalradFe18,axis=1)
#    signalarraysumspatialradFe19 = np.sum(signalradFe19,axis=1)
#    signalarraysumspatialradFe21 = np.sum(signalradFe21,axis=1)
#    #signalarraysumspatialradFe22 = np.sum(signalradFe22,axis=1)
#
#signalradmaxFe24 = np.zeros(len(signalradFe24[0]))
#for iii in range(len(signalradFe24[0])):
#    signalradmaxFe24[iii] = signalradFe24[:,iii].max()
#print('signalradFe24.shape is: ',signalradFe24.shape)
#print('signalradmaxFe24.shape is: ',signalradmaxFe24.shape)
#time = data['time']
#print('Time at timeslice ',timeslice,' is: ',time[timeslice])
#rho = data['rho_poloidal_grid']
#radius = data['radius_grid']*.01
#diff, v = data['anomal_diffusion'], data['anomal_drift']
#nedata = data['electron_density']*10**(-13)
#tedata = data['electron_temperature']*0.001
#tidata = data['proton_temperature']*0.001
#
#diff2, v2 = data2['anomal_diffusion'], data2['anomal_drift']
#rho2 = data2['rho_poloidal_grid']
#diff3, v3 = data3['anomal_diffusion'], data3['anomal_drift']
#rho3 = data3['rho_poloidal_grid']
#
#
#
#signalwithbackground = resultsname.T
#signalwithbackground2 = resultsname2.T
##signalwithbackground3 = resultsname3.T
#
#for kk in range(len(rhosliced)):
#    if kk == 0:
#        dataarray = (np.expand_dims(signalsliced[0:lengthbeforetile],axis=0)).T
#        sigmadataarray = (np.expand_dims(sigmasliced[0:lengthbeforetile],axis=0)).T
#        #statweightdataarray = (np.expand_dims(1/sigmasliced[0:lengthbeforetile],axis=0)).T
#        STRAHLdataarray = (np.expand_dims(signalwithbackground[0:lengthbeforetile],axis=0)).T
#        STRAHLdataarray2 = (np.expand_dims(signalwithbackground2[0:lengthbeforetile],axis=0)).T
#        #STRAHLdataarray3 = (np.expand_dims(signalwithbackground3[0:lengthbeforetile],axis=0)).T
#        #print('dataarray.shape ',dataarray.shape)
#    elif kk > 0:
#        dataarray = np.append(dataarray,(np.expand_dims(signalsliced[kk*lengthbeforetile:(kk+1)*lengthbeforetile],axis=0)).T,axis=1)
#        sigmadataarray = np.append(sigmadataarray,(np.expand_dims(sigmasliced[kk*lengthbeforetile:(kk+1)*lengthbeforetile],axis=0)).T,axis=1)
#        #statweightdataarray = np.append(statweightdataarray,(np.expand_dims(1/sigmasliced[kk*lengthbeforetile:(kk+1)*lengthbeforetile],axis=0)).T,axis=1)
#        STRAHLdataarray = np.append(STRAHLdataarray,(np.expand_dims(signalwithbackground[kk*lengthbeforetile:(kk+1)*lengthbeforetile],axis=0)).T,axis=1)
#        STRAHLdataarray2 = np.append(STRAHLdataarray2,(np.expand_dims(signalwithbackground2[kk*lengthbeforetile:(kk+1)*lengthbeforetile],axis=0)).T,axis=1)
#        #STRAHLdataarray3 = np.append(STRAHLdataarray3,(np.expand_dims(signalwithbackground3[kk*lengthbeforetile:(kk+1)*lengthbeforetile],axis=0)).T,axis=1)
#
#
#
#linewidth = 170
#titlesize = 20
#labelsize = 20
#ticksize = 20
#pointsize = 100
#
#
#plt.figure()
##plt.plot(rho,diff[timeslice,:]*(0.01**2),label='_nolegend_',color='C0')
##plt.scatter(fitparams_name[0,0:12],fitparams_name[2,0:12],s=pointsize,color='C1')
##plt.plot(rho2,diff2[timeslice,:]*(0.01**2),label='_nolegend_',color='C2')
##plt.scatter(fitparams_name2[0,0:12],fitparams_name2[2,0:12],s=pointsize,color='C3')
##plt.plot(rho3,diff3[timeslice,:]*(0.01**2),label='_nolegend_',color='C4')
##plt.scatter(fitparams_name3[0,0:12],fitparams_name3[2,0:12],s=pointsize,color='C5')
#plt.errorbar(fitparams_name[0,0:12],fitparams_name[2,0:12],yerr=fitparams_name[3,0:12],marker='o',markersize=pointsize-90,capsize=5,elinewidth=2,markeredgewidth=2)
#plt.errorbar(fitparams_name2[0,0:12],fitparams_name2[2,0:12],yerr=fitparams_name2[3,0:12],marker='o',markersize=pointsize-90,capsize=5,elinewidth=2,markeredgewidth=2)
##plt.errorbar(fitparams_name3[0,0:12],fitparams_name3[2,0:12],yerr=fitparams_name3[3,0:12],marker='o',markersize=pointsize-90,capsize=5,elinewidth=2,markeredgewidth=2)
##plt.errorbar(fitparams_name4[0,0:12],fitparams_name4[2,0:12],yerr=fitparams_name4[3,0:12],marker='o',markersize=pointsize-90,capsize=5,elinewidth=2,markeredgewidth=2)
##plt.errorbar(fitparams_name5[0,0:12],fitparams_name5[2,0:12],yerr=fitparams_name5[3,0:12],marker='o',markersize=pointsize-90,capsize=5,elinewidth=2,markeredgewidth=2)
#
##plt.title('\n'.join(wrap('Diffusion profile for strahlfile: '+name+'',width=linewidth)),fontsize=titlesize)
#plt.ylabel('Diffusion coefficient (m^2/s)',fontsize=labelsize)
#plt.xlabel('Rho (r/a)',fontsize=labelsize)
##plt.legend(['1','2','3','4','5'],fontsize=labelsize-6)
#plt.xticks(size=ticksize)
#plt.yticks(size=ticksize)
##plt.xlim(0.0,0.85)
#plt.ylim(0.0,3.0)
#
#
#plt.figure()
##plt.plot(rho,v[timeslice,:]*0.01,label='_nolegend_',color='C0')
##plt.scatter(fitparams_name[0,12:24],fitparams_name[2,12:24],s=pointsize,color='C1')
##plt.plot(rho2,v2[timeslice,:]*0.01,label='_nolegend_',color='C2')
##plt.scatter(fitparams_name2[0,12:24],fitparams_name2[2,12:24],s=pointsize,color='C3')
##plt.plot(rho3,v3[timeslice,:]*0.01,label='_nolegend_',color='C4')
##plt.scatter(fitparams_name3[0,12:24],fitparams_name3[2,12:24],s=pointsize,color='C5')
#plt.errorbar(fitparams_name[0,12:24],fitparams_name[2,12:24],yerr=fitparams_name[3,12:24],marker='o',markersize=pointsize-90,capsize=5,elinewidth=2,markeredgewidth=2)
#plt.errorbar(fitparams_name2[0,12:24],fitparams_name2[2,12:24],yerr=fitparams_name2[3,12:24],marker='o',markersize=pointsize-90,capsize=5,elinewidth=2,markeredgewidth=2)
##plt.errorbar(fitparams_name3[0,12:24],fitparams_name3[2,12:24],yerr=fitparams_name3[3,12:24],marker='o',markersize=pointsize-90,capsize=5,elinewidth=2,markeredgewidth=2)
##plt.errorbar(fitparams_name4[0,12:24],fitparams_name4[2,12:24],yerr=fitparams_name4[3,12:24],marker='o',markersize=pointsize-90,capsize=5,elinewidth=2,markeredgewidth=2)
##plt.errorbar(fitparams_name5[0,12:24],fitparams_name5[2,12:24],yerr=fitparams_name5[3,12:24],marker='o',markersize=pointsize-90,capsize=5,elinewidth=2,markeredgewidth=2)
#plt.axhline(y=0,color='black',label='_nolegend_')
##plt.title('\n'.join(wrap('Convective velocity profile for strahlfile: '+name+'',width=linewidth)),fontsize=titlesize)
#plt.ylabel('Convective velocity (m/s)',fontsize=labelsize)
#plt.xlabel('Rho (r/a)',fontsize=labelsize)
##plt.legend(['1','2','3','4','5'],fontsize=labelsize-6)
#plt.xticks(size=ticksize)
#plt.yticks(size=ticksize)
##plt.xlim(0.0,0.85)
#plt.ylim(-15.0,15.0)
#
#
#plt.figure()
#plt.scatter(rho,diff[timeslice,:]*(0.01**2),s=pointsize)
#plt.scatter(fitparams_name[0,0:12],fitparams_name[2,0:12],s=pointsize)
#plt.ylabel('Diffusion coefficient (m^2/s)',fontsize=labelsize)
#plt.xlabel('Rho (r/a)',fontsize=labelsize)
#plt.xticks(size=ticksize)
#plt.yticks(size=ticksize)
#
#plt.figure()
#plt.scatter(rho,v[timeslice,:]*0.01,s=pointsize)
#plt.scatter(fitparams_name[0,12:24],fitparams_name[2,12:24],s=pointsize)
#plt.ylabel('Convective velocity (m/s)',fontsize=labelsize)
#plt.xlabel('Rho (r/a)',fontsize=labelsize)
#plt.xticks(size=ticksize)
#plt.yticks(size=ticksize)
##plt.xlim(0.0,0.9)
##plt.ylim(-2.0,2.0)
#
#
#print('dataarray.shape is ',dataarray.shape)
#print('STRAHLdataarray.shape is ',STRAHLdataarray.shape)
#
#plt.figure()
#plt.contourf(rhosliced,timesliced[:lengthbeforetile],dataarray,levels=[0,1,2,3,4,5,6,7,8])
#colorbar=plt.colorbar()
#colorbar.set_label('Intensity (arb)',fontsize=15)
#plt.xlabel("Rho (r/a)",fontsize=15)
#plt.ylabel("Time (s)",fontsize=15)
##plt.title(title,fontsize=15)
#
#plt.figure()
#plt.contourf(rhosliced,timesliced[:lengthbeforetile],STRAHLdataarray,levels=[0,1,2,3,4,5,6,7,8])
#colorbar=plt.colorbar()
#colorbar.set_label('Intensity (arb)',fontsize=15)
#plt.xlabel("Rho (r/a)",fontsize=15)
#plt.ylabel("Time (s)",fontsize=15)
#
#plt.figure()
#plt.contourf(rhosliced,timesliced[:lengthbeforetile],(1/sigmadataarray)*(dataarray-STRAHLdataarray),levels=[-3,-2,-1,0,1,2,3,4,5,6,7,8])
#colorbar=plt.colorbar()
#colorbar.set_label('Weighted residual (arb)',fontsize=15)
#plt.xlabel("Rho (r/a)",fontsize=15)
#plt.ylabel("Time (s)",fontsize=15)
#
#plt.figure()
#plt.contourf(rhosliced,timesliced[:lengthbeforetile],(1/sigmadataarray)*(dataarray-STRAHLdataarray))
#colorbar=plt.colorbar()
#colorbar.set_label('Weighted residual (arb)',fontsize=15)
#plt.xlabel("Rho (r/a)",fontsize=15)
#plt.ylabel("Time (s)",fontsize=15)
#
#plt.figure()
#plt.contourf(rhosliced,timesliced[:lengthbeforetile],sigmadataarray)
#colorbar=plt.colorbar()
#colorbar.set_label('Sigmas (arb)',fontsize=15)
#plt.xlabel("Rho (r/a)",fontsize=15)
#plt.ylabel("Time (s)",fontsize=15)
#
#x,y = np.meshgrid(rhosliced,timesliced[:lengthbeforetile])
#
#fig  = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#surf=ax.plot_surface(x,y,dataarray,cmap=plt.get_cmap('viridis'),vmin=0,vmax=8.0)#,vmin=0,vmax=10.5)
#cb=plt.colorbar(surf)
#cb.set_label('Intensity (arb)',fontsize=15)
#ax.set_xlabel('Rho (r/a)',fontsize=15)
#ax.set_ylabel('Time (s)',fontsize=15)
#ax.set_zlim(0,8.0)
#
#fig  = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#surf=ax.plot_surface(x,y,STRAHLdataarray,cmap=plt.get_cmap('viridis'),vmin=0,vmax=8.0)#,vmin=0,vmax=10.5)
#cb=plt.colorbar(surf)
#cb.set_label('Intensity (arb)',fontsize=15)
#ax.set_xlabel('Rho (r/a)',fontsize=15)
#ax.set_ylabel('Time (s)',fontsize=15)
#ax.set_zlim(0,8.0)
#
#fig  = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#surf=ax.plot_surface(x,y,(1/sigmadataarray)*(dataarray-STRAHLdataarray),cmap=plt.get_cmap('viridis'),vmin=-3,vmax=8.0)#,vmin=0,vmax=10.5)
#cb=plt.colorbar(surf)
#cb.set_label('Weighted residual (arb)',fontsize=15)
#ax.set_xlabel('Rho (r/a)',fontsize=15)
#ax.set_ylabel('Time (s)',fontsize=15)
#ax.set_zlim(0,8.0)
#
#fig  = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#surf=ax.plot_surface(x,y,sigmadataarray,cmap=plt.get_cmap('viridis'))#,vmin=-3,vmax=8.0)#,vmin=0,vmax=10.5)
#cb=plt.colorbar(surf)
#cb.set_label('Sigmas (arb)',fontsize=15)
#ax.set_xlabel('Rho (r/a)',fontsize=15)
#ax.set_ylabel('Time (s)',fontsize=15)
#ax.set_zlim(0,8.0)
#
#
#
#
#
#
#'''
#fig2 = plt.figure(21)
#gs2 = gridspec.GridSpec(2,1,height_ratios=[2,1])
#
#ax12 = plt.subplot(gs2[0])
#
#print('')
#print('len(resultsnametime) ',len(resultsnametime))
#print('lengthbeforesumsignal ',lengthbeforesumsignal)
#print('len(resultsnamedata) ',len(resultsnamedata))
##print('len(totxicstime) ',len(totxicstime))
##print('len(totxicstimesliced) ',len(totxicstimesliced))
#print('lengthbeforetile ', lengthbeforetile)
#print('len(tottimehexos) ',len(tottimehexos))
#
#fig2.suptitle('Hexos: Fe XXII ~ 11.72 nm',fontsize=15)
#ax12.errorbar(tottimehexos,resultsnamedata[lengthbeforesumsignal:lengthbeforesumsignal+len(tottimehexos)],yerr=resultsnamesigma[lengthbeforesumsignal:lengthbeforesumsignal+len(tottimehexos)], fmt='bo', ms=2)
#ax12.scatter(tottimehexos, resultsname[lengthbeforesumsignal:lengthbeforesumsignal+len(tottimehexos)], color='red')
#ax12.set_ylabel('Intensity (arb)',fontsize=15)
#ax12.ticklabel_format(style='sci',axis='y',scilimits=(-4,4))
#plt.setp(ax12.get_xticklabels(), visible=False)
#
#ax22 = plt.subplot(gs2[1],sharex=ax12)
#ax22.scatter(tottimehexos,(1/resultsnamesigma[lengthbeforesumsignal:lengthbeforesumsignal+len(tottimehexos)])*(resultsnamedata[lengthbeforesumsignal:lengthbeforesumsignal+len(tottimehexos)]-resultsname[lengthbeforesumsignal:lengthbeforesumsignal+len(tottimehexos)]))
#ax22.set_ylabel('Weighted residual',fontsize=15)
#ax22.set_xlabel('Time (s)',fontsize=15)
#ax22.ticklabel_format(style='sci',axis='y',scilimits=(-4,4))
#plt.subplots_adjust(hspace=.0)
#
#newlength = lengthbeforesumsignal + len(tottimehexos)
#
#fig3 = plt.figure(19)
#gs3 = gridspec.GridSpec(2,1,height_ratios=[2,1])
#ax13 = plt.subplot(gs3[0])
#fig3.suptitle('Hexos: Fe XX ~ 12.18 nm',fontsize=15)
#ax13.errorbar(tottimehexos,resultsnamedata[newlength:newlength+len(tottimehexos)],yerr=resultsnamesigma[newlength:newlength+len(tottimehexos)], fmt='bo', ms=2)
#ax13.scatter(tottimehexos, resultsname[newlength:newlength+len(tottimehexos)], color='red')
#ax13.set_ylabel('Intensity (arb)',fontsize=15)
#ax13.ticklabel_format(style='sci',axis='y',scilimits=(-4,4))
#plt.setp(ax13.get_xticklabels(), visible=False)
#
#ax23 = plt.subplot(gs3[1],sharex=ax13)
#ax23.scatter(tottimehexos,(1/resultsnamesigma[newlength:newlength+len(tottimehexos)])*(resultsnamedata[newlength:newlength+len(tottimehexos)]-resultsname[newlength:newlength+len(tottimehexos)]))
#ax23.set_ylabel('Weighted residual',fontsize=15)
#ax23.set_xlabel('Time (s)',fontsize=15)
#ax23.ticklabel_format(style='sci',axis='y',scilimits=(-4,4))
#plt.subplots_adjust(hspace=.0)
#
#newlength = newlength + len(tottimehexos)
#
#fig4 = plt.figure(18)
#gs4 = gridspec.GridSpec(2,1,height_ratios=[2,1])
#ax14 = plt.subplot(gs4[0])
#fig4.suptitle('Hexos: Fe XIX ~ 10.84 nm',fontsize=15)
#ax14.errorbar(tottimehexos,resultsnamedata[newlength:newlength+len(tottimehexos)],yerr=resultsnamesigma[newlength:newlength+len(tottimehexos)], fmt='bo', ms=2)
#ax14.scatter(tottimehexos, resultsname[newlength:newlength+len(tottimehexos)], color='red')
#ax14.set_ylabel('Intensity (arb)',fontsize=15)
#ax14.ticklabel_format(style='sci',axis='y',scilimits=(-4,4))
#plt.setp(ax14.get_xticklabels(), visible=False)
#
#ax24 = plt.subplot(gs4[1],sharex=ax14)
#ax24.scatter(tottimehexos,(1/resultsnamesigma[newlength:newlength+len(tottimehexos)])*(resultsnamedata[newlength:newlength+len(tottimehexos)]-resultsname[newlength:newlength+len(tottimehexos)]))
#ax24.set_ylabel('Weighted residual',fontsize=15)
#ax24.set_xlabel('Time (s)',fontsize=15)
#ax24.ticklabel_format(style='sci',axis='y',scilimits=(-4,4))
#plt.subplots_adjust(hspace=.0)
#'''
#
#
#
#
#
#
#
#
#
#
#
#plt.figure()
#plt.scatter(rho2,diff2[timeslice,:]*(0.01**2),s=pointsize)
#plt.scatter(fitparams_name2[0,0:12],fitparams_name2[2,0:12],s=pointsize)
#plt.ylabel('Diffusion coefficient (m^2/s)',fontsize=labelsize)
#plt.xlabel('Rho (r/a)',fontsize=labelsize)
#plt.xticks(size=ticksize)
#plt.yticks(size=ticksize)
#
#plt.figure()
#plt.scatter(rho2,v2[timeslice,:]*0.01,s=pointsize)
#plt.scatter(fitparams_name2[0,12:24],fitparams_name2[2,12:24],s=pointsize)
#plt.ylabel('Convective velocity (m/s)',fontsize=labelsize)
#plt.xlabel('Rho (r/a)',fontsize=labelsize)
#plt.xticks(size=ticksize)
#plt.yticks(size=ticksize)
##plt.xlim(0.0,0.9)
##plt.ylim(-2.0,2.0)
#
#
#plt.figure()
#plt.contourf(rhosliced,timesliced[:lengthbeforetile],dataarray,levels=[0,1,2,3,4,5,6,7,8])
#colorbar=plt.colorbar()
#colorbar.set_label('Intensity (arb)',fontsize=15)
#plt.xlabel("Rho (r/a)",fontsize=15)
#plt.ylabel("Time (s)",fontsize=15)
##plt.title(title,fontsize=15)
#
#plt.figure()
#plt.contourf(rhosliced,timesliced[:lengthbeforetile],STRAHLdataarray2)
#colorbar=plt.colorbar()
#colorbar.set_label('Intensity (arb)',fontsize=15)
#plt.xlabel("Rho (r/a)",fontsize=15)
#plt.ylabel("Time (s)",fontsize=15)
#
#plt.figure()
#plt.contourf(rhosliced,timesliced[:lengthbeforetile],(1/sigmadataarray)*(dataarray-STRAHLdataarray2),levels=[-3,-2,-1,0,1,2,3,4,5,6,7,8])
#colorbar=plt.colorbar()
#colorbar.set_label('Weighted residual (arb)',fontsize=15)
#plt.xlabel("Rho (r/a)",fontsize=15)
#plt.ylabel("Time (s)",fontsize=15)
#
#plt.figure()
#plt.contourf(rhosliced,timesliced[:lengthbeforetile],(1/sigmadataarray)*(dataarray-STRAHLdataarray2))
#colorbar=plt.colorbar()
#colorbar.set_label('Weighted residual (arb)',fontsize=15)
#plt.xlabel("Rho (r/a)",fontsize=15)
#plt.ylabel("Time (s)",fontsize=15)
#
#x,y = np.meshgrid(rhosliced,timesliced[:lengthbeforetile])
#
#fig  = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#surf=ax.plot_surface(x,y,dataarray,cmap=plt.get_cmap('viridis'),vmin=0,vmax=8.0)#,vmin=0,vmax=10.5)
#cb=plt.colorbar(surf)
#cb.set_label('Intensity (arb)',fontsize=15)
#ax.set_xlabel('Rho (r/a)',fontsize=15)
#ax.set_ylabel('Time (s)',fontsize=15)
#ax.set_zlim(0,8.0)
#
#fig  = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#surf=ax.plot_surface(x,y,STRAHLdataarray2,cmap=plt.get_cmap('viridis'),vmin=0,vmax=8.0)#,vmin=0,vmax=10.5)
#cb=plt.colorbar(surf)
#cb.set_label('Intensity (arb)',fontsize=15)
#ax.set_xlabel('Rho (r/a)',fontsize=15)
#ax.set_ylabel('Time (s)',fontsize=15)
#ax.set_zlim(0,8.0)
#
#fig  = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#surf=ax.plot_surface(x,y,(1/sigmadataarray)*(dataarray-STRAHLdataarray2),cmap=plt.get_cmap('viridis'),vmin=-3,vmax=6.0)#,vmin=0,vmax=10.5)
#cb=plt.colorbar(surf)
#cb.set_label('Weighted residual (arb)',fontsize=15)
#ax.set_xlabel('Rho (r/a)',fontsize=15)
#ax.set_ylabel('Time (s)',fontsize=15)
#ax.set_zlim(0,8.0)
#
#
#
#'''
#fig2 = plt.figure(42)
#gs2 = gridspec.GridSpec(2,1,height_ratios=[2,1])
#
#ax12 = plt.subplot(gs2[0])
#
#fig2.suptitle('Hexos: Fe XXII ~ 11.72 nm',fontsize=15)
#ax12.errorbar(tottimehexos,resultsnamedata2[lengthbeforesumsignal:lengthbeforesumsignal+len(tottimehexos)],yerr=resultsnamesigma2[lengthbeforesumsignal:lengthbeforesumsignal+len(tottimehexos)], fmt='bo', ms=2)
#ax12.scatter(tottimehexos, resultsname2[lengthbeforesumsignal:lengthbeforesumsignal+len(tottimehexos)], color='red')
#ax12.set_ylabel('Intensity (arb)',fontsize=15)
#ax12.ticklabel_format(style='sci',axis='y',scilimits=(-4,4))
#plt.setp(ax12.get_xticklabels(), visible=False)
#
#ax22 = plt.subplot(gs2[1],sharex=ax12)
#ax22.scatter(tottimehexos,(1/resultsnamesigma2[lengthbeforesumsignal:lengthbeforesumsignal+len(tottimehexos)])*(resultsnamedata2[lengthbeforesumsignal:lengthbeforesumsignal+len(tottimehexos)]-resultsname2[lengthbeforesumsignal:lengthbeforesumsignal+len(tottimehexos)]))
#ax22.set_ylabel('Weighted residual',fontsize=15)
#ax22.set_xlabel('Time (s)',fontsize=15)
#ax22.ticklabel_format(style='sci',axis='y',scilimits=(-4,4))
#plt.subplots_adjust(hspace=.0)
#
#newlength = lengthbeforesumsignal + len(tottimehexos)
#
#fig3 = plt.figure(38)
#gs3 = gridspec.GridSpec(2,1,height_ratios=[2,1])
#ax13 = plt.subplot(gs3[0])
#fig3.suptitle('Hexos: Fe XX ~ 12.18 nm',fontsize=15)
#ax13.errorbar(tottimehexos,resultsnamedata2[newlength:newlength+len(tottimehexos)],yerr=resultsnamesigma2[newlength:newlength+len(tottimehexos)], fmt='bo', ms=2)
#ax13.scatter(tottimehexos, resultsname2[newlength:newlength+len(tottimehexos)], color='red')
#ax13.set_ylabel('Intensity (arb)',fontsize=15)
#ax13.ticklabel_format(style='sci',axis='y',scilimits=(-4,4))
#plt.setp(ax13.get_xticklabels(), visible=False)
#
#ax23 = plt.subplot(gs3[1],sharex=ax13)
#ax23.scatter(tottimehexos,(1/resultsnamesigma2[newlength:newlength+len(tottimehexos)])*(resultsnamedata2[newlength:newlength+len(tottimehexos)]-resultsname2[newlength:newlength+len(tottimehexos)]))
#ax23.set_ylabel('Weighted residual',fontsize=15)
#ax23.set_xlabel('Time (s)',fontsize=15)
#ax23.ticklabel_format(style='sci',axis='y',scilimits=(-4,4))
#plt.subplots_adjust(hspace=.0)
#
#newlength = newlength + len(tottimehexos)
#
#fig4 = plt.figure(36)
#gs4 = gridspec.GridSpec(2,1,height_ratios=[2,1])
#ax14 = plt.subplot(gs4[0])
#fig4.suptitle('Hexos: Fe XIX ~ 10.84 nm',fontsize=15)
#ax14.errorbar(tottimehexos,resultsnamedata2[newlength:newlength+len(tottimehexos)],yerr=resultsnamesigma2[newlength:newlength+len(tottimehexos)], fmt='bo', ms=2)
#ax14.scatter(tottimehexos, resultsname2[newlength:newlength+len(tottimehexos)], color='red')
#ax14.set_ylabel('Intensity (arb)',fontsize=15)
#ax14.ticklabel_format(style='sci',axis='y',scilimits=(-4,4))
#plt.setp(ax14.get_xticklabels(), visible=False)
#
#ax24 = plt.subplot(gs4[1],sharex=ax14)
#ax24.scatter(tottimehexos,(1/resultsnamesigma2[newlength:newlength+len(tottimehexos)])*(resultsnamedata2[newlength:newlength+len(tottimehexos)]-resultsname2[newlength:newlength+len(tottimehexos)]))
#ax24.set_ylabel('Weighted residual',fontsize=15)
#ax24.set_xlabel('Time (s)',fontsize=15)
#ax24.ticklabel_format(style='sci',axis='y',scilimits=(-4,4))
#plt.subplots_adjust(hspace=.0)
#
#print(*timesliced)
#'''

