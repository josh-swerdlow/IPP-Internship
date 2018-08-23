# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 11:16:54 2018

@author: petr
"""

import logging
import numpy as np
import mpfit
import raw_data_access
from matplotlib import pyplot as plt
from matplotlib import gridspec
from scipy.io import netcdf
from scipy.stats import gaussian_kde
from scipy.stats import norm
from textwrap import wrap
#import scipy.interpolate

def readstrahlnetcdf(result):
    file = netcdf.netcdf_file(result, 'r')
    signal = file.variables
    #testsignal = file.variables['impurity_density'].copy()
    dimension = file.dimensions
    attribute = file._attributes
    
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
    data = {}
    empty = ()
    for keys, values in signal.items():
        print(keys,'...........................',values.shape)
        if values.shape is empty:
            data[keys] = values.data
        else:
            data[keys] = signal[keys][:].copy()
        #data[keys] = {keys : signal[keys][:].copy()}
    print('')
    print('')
 
    file.close()
    
    return dimension, data, attribute

def linearmodel(coeff
                ,x):
    """
    Evaluate a linear function to the background of line radiation before 
    and after and LBO injection

    Fitting is done as a function of x 
    where x is the time window before LBO and after LBO
    Note that x window is the same for ALL signals (this might be changed later)
    """

    y = np.zeros(len(x))

    for ii in range(len(x)):
        y[ii] = coeff[0]*x[ii] + coeff[1]
        
    return y

def linearmodelresidual(coeff
                        ,x=None
                        ,data=None
                        ,sigma=None):
    """
    Return the residual for the loglinearstrahl funciton.
    This method can be bassed to mir_mpfit.
    """

    yfit = linearmodel(coeff
                       ,x=x)
    
    residual = (data - yfit) / sigma
      
    return {'residual':residual}

def linearmodelfit(x=None
                   ,data=None
                   ,sigma=None
                   ,sigtype=None):
    
    residual_keywords = {'x':x
                         ,'data':data
                         ,'sigma':sigma}
    
    
    coeffinitialguess = np.zeros(2)+100.0
    

    parinfo=[{'step':0.0001,'fixed':True,'limited':[0,0],'limits':[0.0,0.0]} for jj in range(len(coeffinitialguess))]
    
    for ii in range(len(coeffinitialguess)):
        parinfo[ii]['fixed'] = False
        

    parinfoarray = np.zeros(len(parinfo))
    parinfoarrayfixedbool = []
    for ii in range(len(parinfo)):
        parinfoarray[ii] = parinfo[ii]['step']
        parinfoarrayfixedbool.append(parinfo[ii]['fixed'])

    mp = mpfit.mpfit(linearmodelresidual
                     ,coeffinitialguess
                     ,parinfo=parinfo
                     ,residual_keywords=residual_keywords)
    
    
    logging.info(mp.errmsg)
    logging.info('mpfit status:')
    logging.info(mp.statusString())
    
    if mp.perror is None: mp.perror = coeffinitialguess*0
        
    yfit = linearmodel(coeff=mp.params
                       ,x=x)
    
    
    return {'description':'Linear: y[ii+(jj*len(x))] = coeff[0+(jj*2)] - coeff[1+(jj*2)]*x[ii]'
            ,'summedchisquared':mp.fnorm
            ,'time':x
            ,'fit':yfit
            ,'params':mp.params
            ,'perror':mp.perror
            ,'coeffinitial':coeffinitialguess
            ,'minparamstepsize': parinfoarray
            ,'fitstatus':mp.status
            ,'fititerations':mp.niter
            ,'numfunctcalls':mp.nfev
            ,'sigtype':sigtype
            ,'fixedbool': parinfoarrayfixedbool}

def print_results(results):
    
    print('{} {}{}'.format('The fit status is:',results['fitstatus'],'\n'))
    print('{} {}{}'.format('The number of iterations completed:',results['fititerations'],'\n'))
    print('{} {}{}'.format('The number of calls to linearmodel:',results['numfunctcalls'],'\n'))
    print('Results:')
    print('{:>5s}  {:>6s}  {:>4s}  {:>15s}  {:>15s}  {:>6s}'.format(
        'param'
        ,'start value'
        ,'fit'
        ,'error'
        ,'minstepsize'
        ,'fixed'))
    
    for ii in range(len(results['params'])):
        print('{:>5d}  {:>10.4f}  {:>6.8f} +/- {:>4.8f}  {:>4.8f}  {:>6s}'.format(
            ii
            ,results['coeffinitial'][ii]
            ,results['params'][ii]
            ,results['perror'][ii]
            ,results['minparamstepsize'][ii]
            ,str(results['fixedbool'][ii])))
    print('\n','The total chisquared value is: ', results['summedchisquared'],'\n')

def weightedresidual_plot(results
                          ,x=None
                          ,data=None
                          ,sigma=None
                          ,fsize=None
                          ,title=None):
    
    fig = plt.figure(figsize=fsize)
    gs = gridspec.GridSpec(2,1,height_ratios=[2,1])
    ax1 = plt.subplot(gs[0])
    fig.suptitle(title,fontsize=15)
    ax1.errorbar(x, data, yerr=sigma, fmt='bo', ms=2,zorder=0)
    #ax1.errorbar(x, results['fit'],yerr = (x*results['perror'][0]+results['perror'][1]), color='red',zorder=1)
    ax1.errorbar(x, results['fit'], color='red',zorder=1)
    ax1.scatter(x, results['fit'], color='red',s=2,zorder=2)
    ax1.set_ylabel('Intensity (arb)',fontsize=15)
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2 = plt.subplot(gs[1],sharex=ax1)
    ax2.scatter(x,(1/sigma)*(data-results['fit']))
    ax2.set_ylabel('Weighted residual',fontsize=15)
    ax2.set_xlabel('Time (s)',fontsize=15)
    plt.subplots_adjust(hspace=.0)
    #ax2.set_xlim(0,5.0)
    

def gaussianfitandplot_single(x=None
                              ,data=None
                              ,weightedresidualsorted=None
                              ,numofbins=None
                              ,fsize=None
                              ,title=None):
    
    gaussian = gaussian_kde(weightedresidualsorted)
    xvalsforgaussian = np.arange(weightedresidualsorted[0]
                                 ,weightedresidualsorted[-1]
                                 ,(weightedresidualsorted[-1]-weightedresidualsorted[0])/float(numofbins))
    
    yvalsforgaussian = gaussian.evaluate(xvalsforgaussian)
    
    (mutot, sigmatot) = norm.fit(weightedresidualsorted)
    gaussianfitweighttot = norm.pdf(xvalsforgaussian,mutot,sigmatot)
    print('mutot is: ', mutot)
    print('sigmatot is: ', sigmatot)
    
    #extratitle = '\n'+'Mean: '+str(mutot)+'\n'+'Sigma: '+str(sigmatot)
    fig = plt.figure(figsize=fsize)
    gs = gridspec.GridSpec(1,2,height_ratios=[0.5])
    ax1 = plt.subplot(gs[0,0])
    fig.suptitle(title,fontsize=15)
    ax1.errorbar(x,data, fmt='bo', ms=2,zorder=0)
    ax1.set_xlabel('Time (s)',fontsize=15)
    ax1.set_ylabel('Weighted residual',fontsize=15)
    
    ax2 = plt.subplot(gs[0,1])
    ax2.hist(weightedresidualsorted,bins=numofbins,orientation='horizontal',normed=True,color='C2',alpha=0.3)
    ax2.plot(yvalsforgaussian,xvalsforgaussian,'ro:')
    ax2.plot(gaussianfitweighttot,xvalsforgaussian,color='C2',marker='o',linestyle=':')
    plt.subplots_adjust(wspace=.15)
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.text(0.125,0.025,'Mean: '+str(mutot)+'\n'+'Sigma: '+str(sigmatot),color='C2',fontsize=15,transform=ax2.transAxes,zorder=10)
    
    return sigmatot
    
def plottingthecheckplots(experimentID=None
                          ,time=None
                          ,signal=None
                          ,signaltimesum=None
                          ,sigmabackground=None
                          ,wavelength=None
                          ,intensityshift=None
                          ,slicingbackgroundtime=None
                          ,indexofline=None
                          ,halfwindow=None
                          ,startindex=None
                          ,endindex=None
                          ,sigtype=None
                          ,nominalwavelength=None
                          ,linewidth=None
                          ,titlesize=None
                          ,labelsize=None
                          ,ticksize=None
                          ,pointsize=None
                          ,fsize=None):
    
    if (sigtype != 'Ar17' and sigtype != 'qsx'):
        sigmastr = r'$\sigma_{backgrd}=$'+str(sigmabackground)
        plt.figure(figsize=fsize)
        #plt.plot(time,signal)
        plt.errorbar(time,signal,yerr=sigmabackground,ms=pointsize)
        plt.scatter(time[slicingbackgroundtime:],signal[slicingbackgroundtime:],s=pointsize,zorder=3)
        plt.title('\n'.join(wrap('UNSCALED HEXOS data for shot: '+experimentID+' and detector 2 for ' +sigtype+' @ ' +str(nominalwavelength)+' nm with '+ sigmastr,width=linewidth)),fontsize=titlesize)
        plt.ylabel('Intensity (arb)',fontsize=labelsize)
        plt.xlabel('Time (s)',fontsize=labelsize)
        plt.tight_layout()
        
        plt.figure(figsize=fsize)
        #plt.plot(time[slicingbackgroundtime:],signal[slicingbackgroundtime:],color='C1')
        plt.errorbar(time[slicingbackgroundtime:],signal[slicingbackgroundtime:],yerr=sigmabackground,ms=pointsize,color='C1')
        plt.scatter(time[slicingbackgroundtime:],signal[slicingbackgroundtime:],s=pointsize,color='C1')
        plt.title('\n'.join(wrap('Zoomed background on UNSCALED HEXOS data for shot: '+experimentID+' and detector 2 for ' +sigtype+' @ ' +str(nominalwavelength)+' nm with '+sigmastr,width=linewidth)),fontsize=titlesize)
        plt.ylabel('Intensity (arb)',fontsize=labelsize)
        plt.xlabel('Time (s)',fontsize=labelsize)
        plt.tight_layout()
        
        plt.figure(figsize=fsize)
        #plt.plot(time,signal+abs(intensityshift))
        plt.errorbar(time,signal+abs(intensityshift),yerr=sigmabackground,ms=pointsize)
        plt.scatter(time,signal+abs(intensityshift),s=pointsize)
        plt.title('\n'.join(wrap('SCALED HEXOS data for shot: '+experimentID+' and detector 2 for ' +sigtype+' @ ' +str(nominalwavelength)+' nm shift by '+str(abs(intensityshift))+' with '+sigmastr,width=linewidth)),fontsize=titlesize)
        plt.ylabel('Intensity (arb)',fontsize=labelsize)
        plt.xlabel('Time (s)',fontsize=labelsize)
        plt.tight_layout()
        
        plt.figure(figsize=fsize)
        plt.plot(wavelength[startindex:endindex],signaltimesum[startindex:endindex])
        plt.scatter(wavelength[startindex:endindex],signaltimesum[startindex:endindex],s=pointsize)
        plt.scatter(wavelength[indexofline],signaltimesum[indexofline],s=pointsize*5,marker='x')
        plt.scatter(wavelength[indexofline+halfwindow],signaltimesum[indexofline+halfwindow],s=pointsize*5,marker='x')
        plt.scatter(wavelength[indexofline-halfwindow],signaltimesum[indexofline-halfwindow],s=pointsize*5,marker='x')
        plt.title('\n'.join(wrap('HEXOS time integrated data for shot: '+experimentID+' and detector 2 for ' +sigtype+' @ ' +str(nominalwavelength)+' nm',width=linewidth)),fontsize=titlesize)
        plt.ylabel('Intensity (arb)',fontsize=labelsize)
        plt.xlabel('Wavelength (nm)',fontsize=labelsize)
        plt.xticks(size=ticksize)
        plt.yticks(size=ticksize)
        plt.tight_layout()
        
        plt.figure(figsize=fsize)
        plt.plot(wavelength[startindex:endindex],signaltimesum[startindex:endindex])
        plt.scatter(wavelength[startindex:endindex],signaltimesum[startindex:endindex],s=pointsize)
        plt.scatter(wavelength[indexofline],signaltimesum[indexofline],s=pointsize*5,marker='x')
        plt.scatter(wavelength[indexofline+halfwindow],signaltimesum[indexofline+halfwindow],s=pointsize*5,marker='x')
        plt.scatter(wavelength[indexofline-halfwindow],signaltimesum[indexofline-halfwindow],s=pointsize*5,marker='x')
        plt.title('\n'.join(wrap('Zoomed HEXOS time integrated data for shot: '+experimentID+' and detector 2 for ' +sigtype+' @ ' +str(nominalwavelength)+' nm',width=linewidth)),fontsize=titlesize)
        plt.ylabel('Intensity (arb)',fontsize=labelsize)
        plt.xlabel('Wavelength (nm)',fontsize=labelsize)
        plt.xticks(size=ticksize)
        plt.yticks(size=ticksize)
        plt.xlim(nominalwavelength-1,nominalwavelength+1)
        plt.tight_layout()
        
    elif (sigtype == 'Ar17' or sigtype == 'qsx'):
        sigmastr = r'$\sigma_{backgrd}=$'+str(sigmabackground)
        plt.figure(figsize=fsize)
        #plt.plot(time,signal)
        plt.errorbar(time,signal,yerr=sigmabackground,ms=pointsize)
        plt.scatter(time[slicingbackgroundtime:],signal[slicingbackgroundtime:],s=pointsize,zorder=3)
        plt.title('\n'.join(wrap('UNSCALED '+sigtype+' data for shot: '+experimentID+' with '+ sigmastr,width=linewidth)),fontsize=titlesize)
        plt.ylabel('Intensity (arb)',fontsize=labelsize)
        plt.xlabel('Time (s)',fontsize=labelsize)
        plt.tight_layout()
        
        plt.figure(figsize=fsize)
        #plt.plot(time[slicingbackgroundtime:],signal[slicingbackgroundtime:],color='C1')
        plt.errorbar(time[slicingbackgroundtime:],signal[slicingbackgroundtime:],yerr=sigmabackground,ms=pointsize,color='C1')
        plt.scatter(time[slicingbackgroundtime:],signal[slicingbackgroundtime:],s=pointsize,color='C1')
        plt.title('\n'.join(wrap('Zoomed background on UNSCALED '+sigtype+' data for shot: '+experimentID+' with '+sigmastr,width=linewidth)),fontsize=titlesize)
        plt.ylabel('Intensity (arb)',fontsize=labelsize)
        plt.xlabel('Time (s)',fontsize=labelsize)
        plt.tight_layout()
        
        plt.figure(figsize=fsize)
        #plt.plot(time,signal+abs(intensityshift))
        plt.errorbar(time,signal+abs(intensityshift),yerr=sigmabackground,ms=pointsize)
        plt.scatter(time,signal+abs(intensityshift),s=pointsize)
        plt.title('\n'.join(wrap('SCALED '+sigtype+' data for shot: '+experimentID+' shift by '+str(abs(intensityshift))+' with '+sigmastr,width=linewidth)),fontsize=titlesize)
        plt.ylabel('Intensity (arb)',fontsize=labelsize)
        plt.xlabel('Time (s)',fontsize=labelsize)
        plt.tight_layout()
        

    
def rungaussianfit(x=None
                ,data=None
                ,sigma=None
                ,sigtype=None
                ,numofbins=None
                ,linewidth=None
                ,fsize=None):
    
    backgroundresult = linearmodelfit(x=x
                                      ,data=data
                                      ,sigma=sigma
                                      ,sigtype=sigtype)
    
    startstr = 'Start: '+(np.array_str(backgroundresult['coeffinitial'],max_line_width=linewidth))[1:-1]
    resultstr = 'Result: '+(np.array_str(backgroundresult['params'],max_line_width=linewidth))[1:-1]
    perrorstr = 'Perror: '+(np.array_str(backgroundresult['perror'],max_line_width=linewidth))[1:-1]
    sigmastr = r'$\sigma_{backgrd}=$'+str(sigma[0])
    #r'$\sigma_{total} = \sqrt{{\sigma_{backgrd}}^{2} + \alpha^{2} {\sigma_{shot}}^{2}} \/with\/ \alpha = 2.48386581052$'

    title = startstr+'\n'+resultstr+'\n'+perrorstr+'\n'+sigmastr
    
    print_results(backgroundresult)
    
    backgroundweightedresidualFeVII = (1/sigma)*(data-backgroundresult['fit'])
    backgroundweightedresidualFeVIIsorted = np.sort(backgroundweightedresidualFeVII)
    
    
    weightedresidual_plot(backgroundresult
                          ,x=x
                          ,data=data
                          ,sigma=sigma
                          ,fsize=fsize
                          ,title=title)
    
    sigmatot = gaussianfitandplot_single(x=x
                                         ,data=data
                                         ,weightedresidualsorted=backgroundweightedresidualFeVIIsorted
                                         ,numofbins=numofbins
                                         ,fsize=fsize
                                         ,title=title)
    
    return sigmatot

class linearbackgroundfittersinglesignal:

    def linearandexponentialmodel(self
                                  ,coeff
                                  ,x=None):
        """
        Evaluate a linear and exponential function to the background of line radiation before 
        and after and LBO injection

        Fitting is done as a function of x 
        where x is the time window before LBO and after LBO
        Note that x window is the same for ALL signals (this might be changed later)
        """

        y = np.zeros(len(x))

        for ii in range(len(x)):
            y[ii] = coeff[0]*x[ii] + coeff[1] + np.exp(coeff[2])*(np.exp(-1.0*coeff[3]*x[ii]))
            
        return y

    def linearandexponentialresidual(self
                                     ,coeff
                                     ,x=None):
        """
        Return the residual for the linearandexponentialmodel funciton.
        This method is based to mir_mpfit.
        """

        yfit = self.linearandexponentialmodel(coeff=coeff
                                              ,x=x)
        
        residual = (self.data - yfit) / self.sigma
          
        return {'residual':residual}

    def linearandexponentialfit(self
                                ,x=None
                                ,sigtype=None):
        
        residual_keywords = {'x':x}
        
        
        coeffinitialguess = np.zeros(4)#+1.0
        
        coeffinitialguess[0] = -2838.10177048 #1.0
        coeffinitialguess[1] = 17073.66240866
        coeffinitialguess[2] = 11.88970053
        coeffinitialguess[3] = 1.44312172

        parinfo=[{'step':0.0001,'fixed':True,'limited':[0,0],'limits':[0.0,0.0]} for jj in range(len(coeffinitialguess))]
        
        for ii in range(len(coeffinitialguess)):
            parinfo[ii]['fixed'] = False
            
        #parinfo[0]['fixed'] = True
        #parinfo[1]['fixed'] = True

        parinfoarray = np.zeros(len(parinfo))
        parinfoarrayfixedbool = []
        for ii in range(len(parinfo)):
            parinfoarray[ii] = parinfo[ii]['step']
            parinfoarrayfixedbool.append(parinfo[ii]['fixed'])

        mp = mpfit.mpfit(self.linearandexponentialresidual
                         ,coeffinitialguess
                         ,parinfo=parinfo
                         ,residual_keywords=residual_keywords)
        
        
        logging.info(mp.errmsg)
        logging.info('mpfit status:')
        logging.info(mp.statusString())
        
        if mp.perror is None: mp.perror = coeffinitialguess*0
            
        yfit = self.linearandexponentialmodel(coeff=mp.params
                                              ,x=x)
        
        
        return {'description':'Linear & exponential : coeff[0]*x[ii] + coeff[1] + np.exp(coeff[2])*(np.exp(-1.0*coeff[3]*x[ii]))'
                ,'summedchisquared':mp.fnorm
                ,'time':x
                ,'fit':yfit
                ,'params':mp.params
                ,'perror':mp.perror
                ,'coeffinitial':coeffinitialguess
                ,'minparamstepsize': parinfoarray
                ,'fitstatus':mp.status
                ,'fititerations':mp.niter
                ,'numfunctcalls':mp.nfev
                ,'sigtype':sigtype
                ,'fixedbool': parinfoarrayfixedbool}
    
    def linearmodel(self
                    ,coeff
                    ,x=None):
        """
        Evaluate a linear function to the background of line radiation before 
        and after and LBO injection

        Fitting is done as a function of x 
        where x is the time window before LBO and after LBO
        Note that x window is the same for ALL signals (this might be changed later)
        """

        y = np.zeros(len(x))

        for ii in range(len(x)):
            y[ii] = coeff[0]*x[ii] + coeff[1]
            
        return y

    def linearmodelresidual(self
                            ,coeff
                            ,x=None):
        """
        Return the residual for the linearmodel funciton.
        This method is based to mir_mpfit.
        """

        yfit = self.linearmodel(coeff=coeff
                                ,x=x)
        
        residual = (self.data - yfit) / self.sigma
          
        return {'residual':residual}

    def linearmodelfit(self
                       ,x=None
                       ,sigtype=None):
        
        residual_keywords = {'x':x}
        
        
        coeffinitialguess = np.zeros(2)+100.0
        

        parinfo=[{'step':0.0001,'fixed':True,'limited':[0,0],'limits':[0.0,0.0]} for jj in range(len(coeffinitialguess))]
        
        for ii in range(len(coeffinitialguess)):
            parinfo[ii]['fixed'] = False
            

        parinfoarray = np.zeros(len(parinfo))
        parinfoarrayfixedbool = []
        for ii in range(len(parinfo)):
            parinfoarray[ii] = parinfo[ii]['step']
            parinfoarrayfixedbool.append(parinfo[ii]['fixed'])

        mp = mpfit.mpfit(self.linearmodelresidual
                         ,coeffinitialguess
                         ,parinfo=parinfo
                         ,residual_keywords=residual_keywords)
        
        
        logging.info(mp.errmsg)
        logging.info('mpfit status:')
        logging.info(mp.statusString())
        
        if mp.perror is None: mp.perror = coeffinitialguess*0
            
        yfit = self.linearmodel(coeff=mp.params
                                ,x=x)
        
        
        return {'description':'Linear: y[ii] = coeff[0]*x[ii] + coeff[1]'
                ,'summedchisquared':mp.fnorm
                ,'time':x
                #,'lengthsig':lengthsig
                ,'fit':yfit
                ,'params':mp.params
                ,'perror':mp.perror
                ,'coeffinitial':coeffinitialguess
                ,'minparamstepsize': parinfoarray
                ,'fitstatus':mp.status
                ,'fititerations':mp.niter
                ,'numfunctcalls':mp.nfev
                ,'sigtype':sigtype
                ,'fixedbool': parinfoarrayfixedbool}

    def print_results(self
                      ,results):
        
        print('{} {}{}'.format('The fit status is:',results['fitstatus'],'\n'))
        print('{} {}{}'.format('The number of iterations completed:',results['fititerations'],'\n'))
        print('{} {}{}'.format('The number of calls to linearmodel:',results['numfunctcalls'],'\n'))
        print('Results:')
        print('{:>5s}  {:>6s}  {:>4s}  {:>15s}  {:>15s}  {:>6s}'.format(
            'param'
            ,'start value'
            ,'fit'
            ,'error'
            ,'minstepsize'
            ,'fixed'))
        
        for ii in range(len(results['params'])):
            print('{:>5d}  {:>10.4f}  {:>6.8f} +/- {:>4.8f}  {:>4.8f}  {:>6s}'.format(
                ii
                ,results['coeffinitial'][ii]
                ,results['params'][ii]
                ,results['perror'][ii]
                ,results['minparamstepsize'][ii]
                ,str(results['fixedbool'][ii])))
        print('\n','The total chisquared value is: ', results['summedchisquared'],'\n')
        
    def weightedresidual_plot(self
                              ,results
                              ,alltimesig=None
                              ,alldatasig=None
                              ,alldatasigma=None
                              ,sigtype=None
                              ,fsize=None
                              ,title=None):
        
        if sigtype == 'Ar17':
            yerrfit = self.linearandexponentialmodel(coeff=results['perror'],x=self.x)
        else:
            yerrfit = self.linearmodel(coeff=results['perror'],x=self.x)
        
        fig = plt.figure(figsize=fsize)
        gs = gridspec.GridSpec(2,1,height_ratios=[2,1])
        ax1 = plt.subplot(gs[0])
        fig.suptitle(title,fontsize=15)
        ax1.errorbar(alltimesig,alldatasig,yerr=alldatasigma, fmt='bo', ms=2,zorder=0)
        ax1.errorbar(self.x, results['fit'],yerr = yerrfit, color='red',zorder=1)
        ax1.scatter(self.x, results['fit'], color='red',s=2,zorder=2)
        ax1.set_ylabel('Intensity (arb)',fontsize=15)
        plt.setp(ax1.get_xticklabels(), visible=False)
    
        ax2 = plt.subplot(gs[1],sharex=ax1)
        ax2.scatter(self.x,(1/self.sigma)*(self.data-results['fit']))
        ax2.set_ylabel('Weighted residual',fontsize=15)
        ax2.set_xlabel('Time (s)',fontsize=15)
        plt.subplots_adjust(hspace=.0)
        #ax2.set_xlim(0,5.0)
        
    def gaussianfitandplot_all(self
                               ,results
                               ,timelengthbeforeLBO=None
                               ,numofbins=None
                               ,fsize=None
                               ,title=None):
        
        weightedresidual = (1/self.sigma)*(self.data-results['fit'])
        weightedresidualsorted = np.sort(weightedresidual)
        
        gaussian = gaussian_kde(weightedresidualsorted)
        xvalsforgaussian = np.arange(weightedresidualsorted[0]
                                     ,weightedresidualsorted[-1]
                                     ,(weightedresidualsorted[-1]-weightedresidualsorted[0])/float(numofbins))
        
        yvalsforgaussian = gaussian.evaluate(xvalsforgaussian)
        
        xvalsforgaussianbefore = np.arange(np.min(weightedresidual[0:timelengthbeforeLBO])
                                           ,np.max(weightedresidual[0:timelengthbeforeLBO])
                                           ,(np.max(weightedresidual[0:timelengthbeforeLBO])-np.min(weightedresidual[0:timelengthbeforeLBO]))/float(numofbins))
        
        xvalsforgaussianafter = np.arange(np.min(weightedresidual[timelengthbeforeLBO:])
                                          ,np.max(weightedresidual[timelengthbeforeLBO:])
                                          ,(np.max(weightedresidual[timelengthbeforeLBO:])-np.min(weightedresidual[timelengthbeforeLBO:]))/float(numofbins))
        
        (mutot, sigmatot) = norm.fit(weightedresidualsorted)
        gaussianfitweighttot = norm.pdf(xvalsforgaussian,mutot,sigmatot)
        print('mutot is: ', mutot)
        print('sigmatot is: ', sigmatot)
        
        (mubefore, sigmabefore) = norm.fit(weightedresidual[0:timelengthbeforeLBO])
        gaussianfitweightbefore = norm.pdf(xvalsforgaussianbefore,mubefore,sigmabefore)
        print('mubefore is: ', mubefore)
        print('sigmabefore is: ', sigmabefore)            

        (muafter, sigmaafter) = norm.fit(weightedresidual[timelengthbeforeLBO:])
        gaussianfitweightafter = norm.pdf(xvalsforgaussianafter,muafter,sigmaafter)
        print('muafter is: ', muafter)
        print('sigmaafter is: ', sigmaafter) 
        
        fig = plt.figure(figsize=fsize)
        gs = gridspec.GridSpec(6,6)
        ax1 = plt.subplot(gs[2:6,0:4])
        fig.suptitle(title,fontsize=15)
        ax1.scatter(self.x[0:timelengthbeforeLBO],weightedresidual[0:timelengthbeforeLBO],color='C0')
        ax1.scatter(self.x[timelengthbeforeLBO:],weightedresidual[timelengthbeforeLBO:],color='C1')
        ax1.set_ylabel('Weighted residual',fontsize=15)
        ax1.set_xlabel('Time (s)',fontsize=15)
        plt.setp(ax1.get_xticklabels(), visible=True)
    
        ax2 = plt.subplot(gs[2:6,4:6])
        ax2.hist(weightedresidualsorted,bins=numofbins,orientation='horizontal',normed=True,color='C2',alpha=0.3)
        ax2.plot(yvalsforgaussian,xvalsforgaussian,'ro:')
        ax2.plot(gaussianfitweighttot,xvalsforgaussian,color='C2',marker='o',linestyle=':')
        plt.subplots_adjust(wspace=.15)
        plt.setp(ax2.get_yticklabels(), visible=False)
        
        ax3 = plt.subplot(gs[0:2,0:4])
        ax3.hist(weightedresidual[0:timelengthbeforeLBO],bins=numofbins,color='C0',normed=True,alpha=0.3)
        ax3.hist(weightedresidual[timelengthbeforeLBO:],bins=numofbins,color='C1',normed=True,alpha=0.3)
        ax3.plot(xvalsforgaussianbefore,gaussianfitweightbefore,color='C0',marker='o',linestyle=':')
        ax3.plot(xvalsforgaussianafter,gaussianfitweightafter,color='C1',marker='o',linestyle=':')
        plt.subplots_adjust(hspace=.15)
        plt.setp(ax3.get_xticklabels(), visible=False)
        
        ax4 = plt.subplot(gs[0:2,4:6])
        ax4.axis('off')
        ax4.text(0,0.1,'Mean: '+str(mutot)+'\n'+'Sigma: '+str(sigmatot),color='C2',fontsize=15)
        ax4.text(0,0.4,'Mean: '+str(muafter)+'\n'+'Sigma: '+str(sigmaafter),color='C1',fontsize=15)
        ax4.text(0,0.7,'Mean: '+str(mubefore)+'\n'+'Sigma: '+str(sigmabefore),color='C0',fontsize=15)
        
        return sigmatot
            
    def plot_results(self
                     ,results
                     ,alltimesig=None
                     ,alldatasig=None
                     ,alldatasigma=None
                     ,timelengthbeforeLBO=None
                     ,numofbins=None
                     ,fsize=None
                     ,linewidth=None):
        
        startstr = 'Start: '+(np.array_str(results['coeffinitial'],max_line_width=linewidth))[1:-1]
        resultstr = 'Result: '+(np.array_str(results['params'],max_line_width=linewidth))[1:-1]
        perrorstr = 'Perror: '+(np.array_str(results['perror'],max_line_width=linewidth))[1:-1]
        sigmastr = r'$\sigma_{total} = \sqrt{{\sigma_{backgrd}}^{2} + \alpha^{2} {\sigma_{shot}}^{2}} \/with\/ \alpha = 1.98798437$' #+str(alldatasigma[0])

        #title = startstr+'\n'+resultstr+'\n'+perrorstr+'\n'+sigmastr
        title = resultstr+'\n'+perrorstr+'\n'+sigmastr
        shotnumber = '171123036'
        
        sigtype = results['sigtype']
        
        if sigtype == 'Ar17':

            backgroundAr17 = self.linearandexponentialmodel(coeff=results['params'],x=alltimesig)
            #weightedresidualAr17 = (1/self.sigma)*(self.data-results['fit'])
            
            self.weightedresidual_plot(results
                                       ,alltimesig=alltimesig
                                       ,alldatasig=alldatasig
                                       ,alldatasigma=alldatasigma
                                       ,sigtype=sigtype
                                       ,fsize=fsize
                                       ,title=title)
            
            sigmatot = self.gaussianfitandplot_all(results
                                                   ,timelengthbeforeLBO=timelengthbeforeLBO
                                                   ,numofbins=numofbins
                                                   ,fsize=fsize
                                                   ,title=title)
            
            fig = plt.figure(figsize=fsize)
            fig.suptitle(title,fontsize=15)
            #plt.scatter(alltimesig,alldatasig-backgroundAr17, c='blue',s=8)
            plt.errorbar(alltimesig,alldatasig-backgroundAr17,yerr=alldatasigma, fmt='bo', ms=2,zorder=0)
            plt.plot(alltimesig,alldatasig-backgroundAr17, c='blue')
            plt.xlabel('Time (s)',fontsize=15)
            plt.ylabel('Intensity (arb)',fontsize=15)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.xlim(1.0,4.5)
            plt.ylim(-1000,50000)
            
            
            #return sigmatot 
            
                
        elif sigtype == 'qsx':
            
            backgroundqsx = self.linearmodel(coeff=results['params'],x=alltimesig)
            #weightedresidualAr17 = (1/self.sigma)*(self.data-results['fit'])
            
            self.weightedresidual_plot(results
                                       ,alltimesig=alltimesig
                                       ,alldatasig=alldatasig
                                       ,alldatasigma=alldatasigma
                                       ,sigtype=sigtype
                                       ,fsize=fsize
                                       ,title=title)
            
            sigmatot = self.gaussianfitandplot_all(results
                                                   ,timelengthbeforeLBO=timelengthbeforeLBO
                                                   ,numofbins=numofbins
                                                   ,fsize=fsize
                                                   ,title=title)
            
            fig = plt.figure(figsize=fsize)
            fig.suptitle(title,fontsize=15)
            #plt.scatter(alltimesig,alldatasig-backgroundAr17, c='blue',s=8)
            plt.errorbar(alltimesig,alldatasig-backgroundqsx,yerr=alldatasigma, fmt='bo', ms=2,zorder=0)
            plt.plot(alltimesig,alldatasig-backgroundqsx, c='blue')
            plt.xlabel('Time (s)',fontsize=15)
            plt.ylabel('Intensity (arb)',fontsize=15)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.xlim(1.0,4.5)
            plt.ylim(-1000,50000)
            
            
            #return sigmatot 
            
        else:
            
            self.weightedresidual_plot(results
                                       ,alltimesig=alltimesig
                                       ,alldatasig=alldatasig
                                       ,alldatasigma=alldatasigma
                                       ,sigtype=sigtype
                                       ,fsize=fsize
                                       ,title=title)
            
            sigmatot = self.gaussianfitandplot_all(results
                                                   ,timelengthbeforeLBO=timelengthbeforeLBO
                                                   ,numofbins=numofbins
                                                   ,fsize=fsize
                                                   ,title=title)
        
        return sigmatot


    def runlinearmodelfit(self
                          ,experimentID
                          ,sigtype
                          ,checkplots=False):

        timetostartbeforeLBOAr17 = 1.85#1.25
        
        timetostartbeforeLBO = 1.5
        timetoendbeforeLBO = 2.05
        timetostartafterLBO = 3.75
        timetoendafterLBO = 3.95
        
        #sigmafactor = 4.1107354312
        #sigmafactor = 1.0
        
        numofbins = 25 #for histogram and gaussian fitting
        
        linewidth = 80
        titlesize = 15
        labelsize = 15
        ticksize = 15
        pointsize = 20
        
        fsize=(10.0,9.0)
        
        if (sigtype != 'Ar17' and sigtype != 'qsx'):
            totsignalHexos2 = raw_data_access.read_hexos_data(experimentID,2)
            
            wavelengthHexos2 = totsignalHexos2['wavelength'] #wavelength is stored LONG to SHORT
            shottimeHexos2 = totsignalHexos2['shottime']
            signal2timesum = np.sum(totsignalHexos2['signal'],axis=0)
            
            halfwindow = 2
            
            indexstartbeforeLBOHexos = (np.abs(shottimeHexos2 - timetostartbeforeLBO)).argmin()
            indexendbeforeLBOHexos = (np.abs(shottimeHexos2 - timetoendbeforeLBO)).argmin()
            indexstartafterLBOHexos = (np.abs(shottimeHexos2 - timetostartafterLBO)).argmin()
            indexendafterLBOHexos = (np.abs(shottimeHexos2 - timetoendafterLBO)).argmin()
            
            timewindowbeforeLBOHexos = shottimeHexos2[indexstartbeforeLBOHexos:indexendbeforeLBOHexos]
            timewindowbackgroundHexos = np.append(timewindowbeforeLBOHexos,shottimeHexos2[indexstartafterLBOHexos:indexendafterLBOHexos])
            
            timelengthbeforeLBO = len(timewindowbeforeLBOHexos)
            alltimesig = shottimeHexos2
            
            numberofbackgroundtimepoints = 1000
            slicingbackgroundtime = int(-1*numberofbackgroundtimepoints)
            
            
        
        if sigtype == 'Ar17':

            #raw_data_access.getImages returns ---> {'image':image_array, 'time':time_array, 'expo_period':expo_period, 'expo_time':expo_time}
            dataAr17 = raw_data_access.getImages('ar17',experimentID,showimage=False)
                    
            #time_arrayAr17=np.loadtxt('C:/Users/petr/Desktop/171123036ar17datatime.csv',delimiter=',')
            #totalsumImarrayAr17=np.loadtxt('C:/Users/petr/Desktop/171123036ar17data.csv',delimiter=',')
            
            time_arrayAr17 = dataAr17['time']
            
            #ar17timesumImarray = np.sum(dataAr17['image'],axis=0)
            #horsumImarray = np.sum(dataAr17['image'],axis=1)
            vertsumImarrayAr17 = np.sum(dataAr17['image'],axis=2)
            totalsumImarrayAr17 = np.sum(vertsumImarrayAr17,axis=1)
            
            indexstartbeforeLBOAr17 = (np.abs(time_arrayAr17 - timetostartbeforeLBOAr17)).argmin()
            indexendbeforeLBOAr17 = (np.abs(time_arrayAr17 - timetoendbeforeLBO)).argmin()
            indexstartafterLBOAr17 = (np.abs(time_arrayAr17 - timetostartafterLBO)).argmin()
            indexendafterLBOAr17 = (np.abs(time_arrayAr17 - timetoendafterLBO)).argmin()
            
            timewindowbeforeLBOAr17 = time_arrayAr17[indexstartbeforeLBOAr17:indexendbeforeLBOAr17]
            timewindowbackgroundAr17 = np.append(timewindowbeforeLBOAr17,time_arrayAr17[indexstartafterLBOAr17:indexendafterLBOAr17])
            backgrounddatafortimewindowAr17 = np.append(totalsumImarrayAr17[indexstartbeforeLBOAr17:indexendbeforeLBOAr17],totalsumImarrayAr17[indexstartafterLBOAr17:indexendafterLBOAr17])
            
            
            numberofbackgroundtimepoints = 500
            slicingbackgroundtime = int(-1*numberofbackgroundtimepoints)
            
            intensityshiftAr17 = np.sum(totalsumImarrayAr17[slicingbackgroundtime:]) / len(totalsumImarrayAr17[slicingbackgroundtime:])
            
            x = time_arrayAr17[slicingbackgroundtime:]
            data = totalsumImarrayAr17[slicingbackgroundtime:]
            sigmaB = 0.178614366348#1.0
            sigma = np.zeros(len(x)) + sigmaB 
            
            sigmabackground = rungaussianfit(x=x
                                             ,data=data
                                             ,sigma=sigma
                                             ,sigtype=sigtype
                                             ,numofbins=numofbins
                                             ,linewidth=linewidth
                                             ,fsize=fsize)
            
            startindex = 1
            endindex = -1
            if checkplots == True:
                #print('np.where(signal2timesum<(-0.2*10.0**8.0)) ',*np.where(signal2timesum<(-0.2*10.0**8.0)))
                #print('len(signal2timesum)',len(signal2timesum))
                plottingthecheckplots(experimentID=experimentID
                                      ,time=time_arrayAr17
                                      ,signal=totalsumImarrayAr17
                                      ,signaltimesum=None#signal2timesum
                                      ,sigmabackground=sigmaB
                                      ,wavelength=None #wavelengthHexos2
                                      ,intensityshift=intensityshiftAr17
                                      ,slicingbackgroundtime=slicingbackgroundtime
                                      ,indexofline=None#indexofFeVII
                                      ,halfwindow=None#halfwindow
                                      ,startindex=startindex
                                      ,endindex=endindex
                                      ,sigtype=sigtype
                                      ,nominalwavelength=None#nominalwavelength
                                      ,linewidth=linewidth
                                      ,titlesize=titlesize
                                      ,labelsize=labelsize
                                      ,ticksize=ticksize
                                      ,pointsize=pointsize
                                      ,fsize=fsize)
            
            
            
            #sigmaAr17 = np.sqrt(abs(backgrounddatafortimewindowAr17))
            
            timelengthbeforeLBO = len(timewindowbeforeLBOAr17)
            alltimesig = time_arrayAr17
            #alldatasig = totalsumImarrayAr17
            #alldatasigma = np.sqrt(abs(alldatasig))
            
            
            self.x = timewindowbackgroundAr17
    
            self.data = backgrounddatafortimewindowAr17
            
            alpha = 2.70237130428#3.17014293-np.sqrt(0.178614366348)#0.0#3.17014293#1.0
            self.sigma = np.sqrt(sigmaB**2.0 + alpha**2.0*backgrounddatafortimewindowAr17)
            #self.sigma = np.sqrt(alpha**2.0*backgrounddatafortimewindowAr17)
            
            alldatasig = totalsumImarrayAr17
            alldatasigma = np.sqrt(sigmaB**2.0 + alpha**2.0*alldatasig)
            #alldatasigma = np.sqrt(alpha**2.0*alldatasig)
            
            
            
        if sigtype == 'qsx':
            #raw_data_access.getImages returns ---> {'image':image_array, 'time':time_array, 'expo_period':expo_period, 'expo_time':expo_time}
            dataqsx = raw_data_access.getImages('qsx',experimentID,showimage=False)
                      
            time_arrayqsx = dataqsx['time']
  
            vertsumImarrayqsx = np.sum(dataqsx['image'],axis=2)
            totalsumImarrayqsx = np.sum(vertsumImarrayqsx,axis=1)
            
            indexstartbeforeLBOqsx = (np.abs(time_arrayqsx - timetostartbeforeLBOAr17)).argmin()
            indexendbeforeLBOqsx = (np.abs(time_arrayqsx - timetoendbeforeLBO)).argmin()
            indexstartafterLBOqsx = (np.abs(time_arrayqsx - timetostartafterLBO)).argmin()
            indexendafterLBOqsx = (np.abs(time_arrayqsx - timetoendafterLBO)).argmin()
    
            timewindowbeforeLBOqsx = time_arrayqsx[indexstartbeforeLBOqsx:indexendbeforeLBOqsx]
            timewindowbackgroundqsx = np.append(timewindowbeforeLBOqsx,time_arrayqsx[indexstartafterLBOqsx:indexendafterLBOqsx])
            backgrounddatafortimewindowqsx = np.append(totalsumImarrayqsx[indexstartbeforeLBOqsx:indexendbeforeLBOqsx],totalsumImarrayqsx[indexstartafterLBOqsx:indexendafterLBOqsx])
            
            
            numberofbackgroundtimepoints = 500
            slicingbackgroundtime = int(-1*numberofbackgroundtimepoints)
            
            intensityshiftqsx = np.sum(totalsumImarrayqsx[slicingbackgroundtime:]) / len(totalsumImarrayqsx[slicingbackgroundtime:])
            
            x = time_arrayqsx[slicingbackgroundtime:]
            data = totalsumImarrayqsx[slicingbackgroundtime:]
            sigmaB = 0.295670769342#1.0#0.178614366348#1.0
            sigma = np.zeros(len(x)) + sigmaB 
            
            sigmabackground = rungaussianfit(x=x
                                             ,data=data
                                             ,sigma=sigma
                                             ,sigtype=sigtype
                                             ,numofbins=numofbins
                                             ,linewidth=linewidth
                                             ,fsize=fsize)
            
            
            startindex = 1
            endindex = -1
            if checkplots == True:
                #print('np.where(signal2timesum<(-0.2*10.0**8.0)) ',*np.where(signal2timesum<(-0.2*10.0**8.0)))
                #print('len(signal2timesum)',len(signal2timesum))
                plottingthecheckplots(experimentID=experimentID
                                      ,time=time_arrayqsx
                                      ,signal=totalsumImarrayqsx
                                      ,signaltimesum=None#signal2timesum
                                      ,sigmabackground=sigmaB
                                      ,wavelength=None #wavelengthHexos2
                                      ,intensityshift=intensityshiftqsx
                                      ,slicingbackgroundtime=slicingbackgroundtime
                                      ,indexofline=None#indexofFeVII
                                      ,halfwindow=None#halfwindow
                                      ,startindex=startindex
                                      ,endindex=endindex
                                      ,sigtype=sigtype
                                      ,nominalwavelength=None#nominalwavelength
                                      ,linewidth=linewidth
                                      ,titlesize=titlesize
                                      ,labelsize=labelsize
                                      ,ticksize=ticksize
                                      ,pointsize=pointsize
                                      ,fsize=fsize)
            
            
            
            #sigmaqsx = np.sqrt(abs(backgrounddatafortimewindowqsx))
            
            timelengthbeforeLBO = len(timewindowbeforeLBOqsx)
            alltimesig = time_arrayqsx
            #alldatasig = totalsumImarrayqsx
            #alldatasigma = np.sqrt(abs(alldatasig))
                                 
            self.x = timewindowbackgroundqsx
    
            self.data = backgrounddatafortimewindowqsx
            
            alpha = 1.98798437#0.0#2.70237130428#1.0
            self.sigma = np.sqrt(sigmaB**2.0 + alpha**2.0*backgrounddatafortimewindowqsx)
            #self.sigma = np.sqrt(alpha**2.0*backgrounddatafortimewindowAr17)
            
            alldatasig = totalsumImarrayqsx
            alldatasigma = np.sqrt(sigmaB**2.0 + alpha**2.0*alldatasig)
            #alldatasigma = np.sqrt(alpha**2.0*alldatasig)

            
        if sigtype == 'FeVII':
            nominalwavelength = 17.86
            
            indexofFeVII = (np.abs(wavelengthHexos2 - nominalwavelength)).argmin()
            #Summing over wavelength range of Iron line
            signalFeVII = np.sum(totsignalHexos2['signal'][:,indexofFeVII-halfwindow:indexofFeVII+halfwindow+1],axis=1)
            #Using background last 1000 time points in summed wavelength iron line signal to calculate average shift
            intensityshiftFeVII = np.sum(signalFeVII[slicingbackgroundtime:]) / len(signalFeVII[slicingbackgroundtime:])
            
            x = shottimeHexos2[slicingbackgroundtime:]
            data = signalFeVII[slicingbackgroundtime:]
            sigmaB = 1.0#154.690050257
            sigma = np.zeros(len(x)) + sigmaB #np.sqrt(abs(data))#
            
            sigmabackground = rungaussianfit(x=x
                                             ,data=data
                                             ,sigma=sigma
                                             ,sigtype=sigtype
                                             ,numofbins=numofbins
                                             ,linewidth=linewidth
                                             ,fsize=fsize)
            
            startindex = 1
            endindex = -1
            if checkplots == True:
                #print('np.where(signal2timesum<(-0.2*10.0**8.0)) ',*np.where(signal2timesum<(-0.2*10.0**8.0)))
                #print('len(signal2timesum)',len(signal2timesum))
                plottingthecheckplots(experimentID=experimentID
                                      ,time=shottimeHexos2
                                      ,signal=signalFeVII
                                      ,signaltimesum=signal2timesum
                                      ,sigmabackground=sigmaB
                                      ,wavelength=wavelengthHexos2
                                      ,intensityshift=intensityshiftFeVII
                                      ,slicingbackgroundtime=slicingbackgroundtime
                                      ,indexofline=indexofFeVII
                                      ,halfwindow=halfwindow
                                      ,startindex=startindex
                                      ,endindex=endindex
                                      ,sigtype=sigtype
                                      ,nominalwavelength=nominalwavelength
                                      ,linewidth=linewidth
                                      ,titlesize=titlesize
                                      ,labelsize=labelsize
                                      ,ticksize=ticksize
                                      ,pointsize=pointsize
                                      ,fsize=fsize)


            signalFeVII = signalFeVII+abs(intensityshiftFeVII)
            
            backgrounddatafortimewindowFeVII = np.append(signalFeVII[indexstartbeforeLBOHexos:indexendbeforeLBOHexos],signalFeVII[indexstartafterLBOHexos:indexendafterLBOHexos])
            
            #sigmaFeVII = sigmafactor*np.sqrt(abs(backgrounddatafortimewindowFeVII))
            
            
            self.x = timewindowbackgroundHexos
    
            self.data = backgrounddatafortimewindowFeVII
            
            alpha = 0.0#2.48386581052
            self.sigma = np.sqrt(sigmaB**2.0 + alpha**2.0*backgrounddatafortimewindowFeVII)
            
            alldatasig = signalFeVII
            alldatasigma = np.sqrt(sigmaB**2.0 + alpha**2.0*alldatasig)#sigmafactor*np.sqrt(abs(alldatasig))            
            
               
        if sigtype == 'FeXIII':
            nominalwavelength = 20.37
            indexofFeXIII = (np.abs(wavelengthHexos2 - nominalwavelength)).argmin()
            signalFeXIII = np.sum(totsignalHexos2['signal'][:,indexofFeXIII-halfwindow:indexofFeXIII+halfwindow+1],axis=1)
            intensityshiftFeXIII = np.sum(signalFeXIII[slicingbackgroundtime:]) / len(signalFeXIII[slicingbackgroundtime:])
            
            x = shottimeHexos2[slicingbackgroundtime:]
            data = signalFeXIII[slicingbackgroundtime:]
            sigmaB = 155.043722909#1.0
            sigma = np.zeros(len(x)) + sigmaB #np.sqrt(abs(data))#
            
            sigmabackground = rungaussianfit(x=x
                                             ,data=data
                                             ,sigma=sigma
                                             ,sigtype=sigtype
                                             ,numofbins=numofbins
                                             ,linewidth=linewidth
                                             ,fsize=fsize)
            
            startindex = 1
            endindex = -1
            if checkplots == True:
                #print('np.where(signal2timesum<(-0.2*10.0**8.0)) ',*np.where(signal2timesum<(-0.2*10.0**8.0)))
                #print('len(signal2timesum)',len(signal2timesum))
                plottingthecheckplots(experimentID=experimentID
                                      ,time=shottimeHexos2
                                      ,signal=signalFeXIII
                                      ,signaltimesum=signal2timesum
                                      ,sigmabackground=sigmaB
                                      ,wavelength=wavelengthHexos2
                                      ,intensityshift=intensityshiftFeXIII
                                      ,slicingbackgroundtime=slicingbackgroundtime
                                      ,indexofline=indexofFeXIII
                                      ,halfwindow=halfwindow
                                      ,startindex=startindex
                                      ,endindex=endindex
                                      ,sigtype=sigtype
                                      ,nominalwavelength=nominalwavelength
                                      ,linewidth=linewidth
                                      ,titlesize=titlesize
                                      ,labelsize=labelsize
                                      ,ticksize=ticksize
                                      ,pointsize=pointsize
                                      ,fsize=fsize)
            
            signalFeXIII = signalFeXIII+abs(intensityshiftFeXIII)
            
            backgrounddatafortimewindowFeXIII = np.append(signalFeXIII[indexstartbeforeLBOHexos:indexendbeforeLBOHexos],signalFeXIII[indexstartafterLBOHexos:indexendafterLBOHexos])
            
            #sigmaFeXIII = sigmafactor*np.sqrt(abs(backgrounddatafortimewindowFeXIII))
            
            self.x = timewindowbackgroundHexos
    
            self.data = backgrounddatafortimewindowFeXIII
    
            alpha = 0.961643528038#0.0
            self.sigma = np.sqrt(sigmaB**2.0 + alpha**2.0*backgrounddatafortimewindowFeXIII)
            
            alldatasig = signalFeXIII
            alldatasigma = np.sqrt(sigmaB**2.0 + alpha**2.0*alldatasig)#sigmafactor*np.sqrt(abs(alldatasig))
            
        if sigtype == 'FeXIX':
            nominalwavelength = 10.84
            indexofFeXIX = (np.abs(wavelengthHexos2 - nominalwavelength)).argmin()
            signalFeXIX = np.sum(totsignalHexos2['signal'][:,indexofFeXIX-halfwindow:indexofFeXIX+halfwindow+1],axis=1)
            intensityshiftFeXIX = np.sum(signalFeXIX[slicingbackgroundtime:]) / len(signalFeXIX[slicingbackgroundtime:])
            
            x = shottimeHexos2[slicingbackgroundtime:]
            data = signalFeXIX[slicingbackgroundtime:]
            sigmaB = 151.809466853#1.0
            sigma = np.zeros(len(x)) + sigmaB #np.sqrt(abs(data))#
            
            sigmabackground = rungaussianfit(x=x
                                             ,data=data
                                             ,sigma=sigma
                                             ,sigtype=sigtype
                                             ,numofbins=numofbins
                                             ,linewidth=linewidth
                                             ,fsize=fsize)
            
            startindex = 1
            endindex = -1
            if checkplots == True:
                #print('np.where(signal2timesum<(-0.2*10.0**8.0)) ',*np.where(signal2timesum<(-0.2*10.0**8.0)))
                #print('len(signal2timesum)',len(signal2timesum))
                plottingthecheckplots(experimentID=experimentID
                                      ,time=shottimeHexos2
                                      ,signal=signalFeXIX
                                      ,signaltimesum=signal2timesum
                                      ,sigmabackground=sigmaB
                                      ,wavelength=wavelengthHexos2
                                      ,intensityshift=intensityshiftFeXIX
                                      ,slicingbackgroundtime=slicingbackgroundtime
                                      ,indexofline=indexofFeXIX
                                      ,halfwindow=halfwindow
                                      ,startindex=startindex
                                      ,endindex=endindex
                                      ,sigtype=sigtype
                                      ,nominalwavelength=nominalwavelength
                                      ,linewidth=linewidth
                                      ,titlesize=titlesize
                                      ,labelsize=labelsize
                                      ,ticksize=ticksize
                                      ,pointsize=pointsize
                                      ,fsize=fsize)
            
            
            signalFeXIX = signalFeXIX+abs(intensityshiftFeXIX)
            
            backgrounddatafortimewindowFeXIX = np.append(signalFeXIX[indexstartbeforeLBOHexos:indexendbeforeLBOHexos],signalFeXIX[indexstartafterLBOHexos:indexendafterLBOHexos])
            
            #sigmaFeXIX = sigmafactor*np.sqrt(abs(backgrounddatafortimewindowFeXIX))
            
            self.x = timewindowbackgroundHexos
    
            self.data = backgrounddatafortimewindowFeXIX
            
            alpha = 2.35933640913#0.0
            self.sigma = np.sqrt(sigmaB**2.0 + alpha**2.0*backgrounddatafortimewindowFeXIX)
            
            alldatasig = signalFeXIX
            alldatasigma = np.sqrt(sigmaB**2.0 + alpha**2.0*alldatasig)#sigmafactor*np.sqrt(abs(alldatasig))
            
        if sigtype == 'FeXX':
            nominalwavelength = 12.18
            indexofFeXX = (np.abs(wavelengthHexos2 - nominalwavelength)).argmin()
            signalFeXX = np.sum(totsignalHexos2['signal'][:,indexofFeXX-halfwindow:indexofFeXX+halfwindow+1],axis=1)
            intensityshiftFeXX = np.sum(signalFeXX[slicingbackgroundtime:]) / len(signalFeXX[slicingbackgroundtime:])
            
            x = shottimeHexos2[slicingbackgroundtime:]
            data = signalFeXX[slicingbackgroundtime:]
            sigmaB = 154.44283169#1.0
            sigma = np.zeros(len(x)) + sigmaB #np.sqrt(abs(data))#
            
            sigmabackground = rungaussianfit(x=x
                                             ,data=data
                                             ,sigma=sigma
                                             ,sigtype=sigtype
                                             ,numofbins=numofbins
                                             ,linewidth=linewidth
                                             ,fsize=fsize)
            
            startindex = 1
            endindex = -1
            if checkplots == True:
                #print('np.where(signal2timesum<(-0.2*10.0**8.0)) ',*np.where(signal2timesum<(-0.2*10.0**8.0)))
                #print('len(signal2timesum)',len(signal2timesum))
                plottingthecheckplots(experimentID=experimentID
                                      ,time=shottimeHexos2
                                      ,signal=signalFeXX
                                      ,signaltimesum=signal2timesum
                                      ,sigmabackground=sigmaB
                                      ,wavelength=wavelengthHexos2
                                      ,intensityshift=intensityshiftFeXX
                                      ,slicingbackgroundtime=slicingbackgroundtime
                                      ,indexofline=indexofFeXX
                                      ,halfwindow=halfwindow
                                      ,startindex=startindex
                                      ,endindex=endindex
                                      ,sigtype=sigtype
                                      ,nominalwavelength=nominalwavelength
                                      ,linewidth=linewidth
                                      ,titlesize=titlesize
                                      ,labelsize=labelsize
                                      ,ticksize=ticksize
                                      ,pointsize=pointsize
                                      ,fsize=fsize)
            
            signalFeXX = signalFeXX+abs(intensityshiftFeXX)
            
            backgrounddatafortimewindowFeXX = np.append(signalFeXX[indexstartbeforeLBOHexos:indexendbeforeLBOHexos],signalFeXX[indexstartafterLBOHexos:indexendafterLBOHexos])
            
            #sigmaFeXX = sigmafactor*np.sqrt(abs(backgrounddatafortimewindowFeXX))
            
            self.x = timewindowbackgroundHexos
    
            self.data = backgrounddatafortimewindowFeXX
    
            alpha = 3.52902275406#0.0
            self.sigma = np.sqrt(sigmaB**2.0 + alpha**2.0*backgrounddatafortimewindowFeXX)
            
            alldatasig = signalFeXX
            alldatasigma = np.sqrt(sigmaB**2.0 + alpha**2.0*alldatasig)#sigmafactor*np.sqrt(abs(alldatasig))

        if sigtype == 'FeXXII':
            nominalwavelength = 11.72
            indexofFeXXII = (np.abs(wavelengthHexos2 - nominalwavelength)).argmin()
            signalFeXXII = np.sum(totsignalHexos2['signal'][:,indexofFeXXII-halfwindow:indexofFeXXII+halfwindow+1],axis=1)
            intensityshiftFeXXII = np.sum(signalFeXXII[slicingbackgroundtime:]) / len(signalFeXXII[slicingbackgroundtime:])
            
            x = shottimeHexos2[slicingbackgroundtime:]
            data = signalFeXXII[slicingbackgroundtime:]
            sigmaB = 154.463064814#1.0
            sigma = np.zeros(len(x)) + sigmaB #np.sqrt(abs(data))#
            
            sigmabackground = rungaussianfit(x=x
                                             ,data=data
                                             ,sigma=sigma
                                             ,sigtype=sigtype
                                             ,numofbins=numofbins
                                             ,linewidth=linewidth
                                             ,fsize=fsize)
            
            startindex = 1
            endindex = -1
            if checkplots == True:
                #print('np.where(signal2timesum<(-0.2*10.0**8.0)) ',*np.where(signal2timesum<(-0.2*10.0**8.0)))
                #print('len(signal2timesum)',len(signal2timesum))
                plottingthecheckplots(experimentID=experimentID
                                      ,time=shottimeHexos2
                                      ,signal=signalFeXXII
                                      ,signaltimesum=signal2timesum
                                      ,sigmabackground=sigmaB
                                      ,wavelength=wavelengthHexos2
                                      ,intensityshift=intensityshiftFeXXII
                                      ,slicingbackgroundtime=slicingbackgroundtime
                                      ,indexofline=indexofFeXXII
                                      ,halfwindow=halfwindow
                                      ,startindex=startindex
                                      ,endindex=endindex
                                      ,sigtype=sigtype
                                      ,nominalwavelength=nominalwavelength
                                      ,linewidth=linewidth
                                      ,titlesize=titlesize
                                      ,labelsize=labelsize
                                      ,ticksize=ticksize
                                      ,pointsize=pointsize
                                      ,fsize=fsize)
            
            signalFeXXII = signalFeXXII+abs(intensityshiftFeXXII)
            
            backgrounddatafortimewindowFeXXII = np.append(signalFeXXII[indexstartbeforeLBOHexos:indexendbeforeLBOHexos],signalFeXXII[indexstartafterLBOHexos:indexendafterLBOHexos])
            
            #sigmaFeXXII = sigmafactor*np.sqrt(abs(backgrounddatafortimewindowFeXXII))
            
            self.x = timewindowbackgroundHexos
    
            self.data = backgrounddatafortimewindowFeXXII
    
            alpha = 5.5269293208#0.0
            self.sigma = np.sqrt(sigmaB**2.0 + alpha**2.0*backgrounddatafortimewindowFeXXII)
            
            alldatasig = signalFeXXII
            alldatasigma = np.sqrt(sigmaB**2.0 + alpha**2.0*alldatasig)#sigmafactor*np.sqrt(abs(alldatasig))
        
        if sigtype == 'FeXXIII':
            nominalwavelength = 13.27
            indexofFeXXIII = (np.abs(wavelengthHexos2 - nominalwavelength)).argmin()
            signalFeXXIII = np.sum(totsignalHexos2['signal'][:,indexofFeXXIII-halfwindow:indexofFeXXIII+halfwindow+1],axis=1)
            intensityshiftFeXXIII = np.sum(signalFeXXIII[-slicingbackgroundtime:]) / len(signalFeXXIII[slicingbackgroundtime:])
            
            x = shottimeHexos2[slicingbackgroundtime:]
            data = signalFeXXIII[slicingbackgroundtime:]
            sigmaB = 1.0#154.690050257
            sigma = np.zeros(len(x)) + sigmaB #np.sqrt(abs(data))#
            
            sigmabackground = rungaussianfit(x=x
                                             ,data=data
                                             ,sigma=sigma
                                             ,sigtype=sigtype
                                             ,numofbins=numofbins
                                             ,linewidth=linewidth
                                             ,fsize=fsize)
            
            startindex = 1
            endindex = -1
            if checkplots == True:
                #print('np.where(signal2timesum<(-0.2*10.0**8.0)) ',*np.where(signal2timesum<(-0.2*10.0**8.0)))
                #print('len(signal2timesum)',len(signal2timesum))
                plottingthecheckplots(experimentID=experimentID
                                      ,time=shottimeHexos2
                                      ,signal=signalFeXXIII
                                      ,signaltimesum=signal2timesum
                                      ,sigmabackground=sigmaB
                                      ,wavelength=wavelengthHexos2
                                      ,intensityshift=intensityshiftFeXXIII
                                      ,slicingbackgroundtime=slicingbackgroundtime
                                      ,indexofline=indexofFeXXIII
                                      ,halfwindow=halfwindow
                                      ,startindex=startindex
                                      ,endindex=endindex
                                      ,sigtype=sigtype
                                      ,nominalwavelength=nominalwavelength
                                      ,linewidth=linewidth
                                      ,titlesize=titlesize
                                      ,labelsize=labelsize
                                      ,ticksize=ticksize
                                      ,pointsize=pointsize
                                      ,fsize=fsize)            
            
            signalFeXXIII = signalFeXXIII+abs(intensityshiftFeXXIII)
            
            backgrounddatafortimewindowFeXXIII = np.append(signalFeXXIII[indexstartbeforeLBOHexos:indexendbeforeLBOHexos],signalFeXXIII[indexstartafterLBOHexos:indexendafterLBOHexos])
            
            #sigmaFeXXIII = sigmafactor*np.sqrt(abs(backgrounddatafortimewindowFeXXIII))
            
            self.x = timewindowbackgroundHexos
    
            self.data = backgrounddatafortimewindowFeXXIII
    
            alpha = 0.0#2.48386581052
            self.sigma = np.sqrt(sigmaB**2.0 + alpha**2.0*backgrounddatafortimewindowFeXXIII)
            
            alldatasig = signalFeXXIII
            alldatasigma = np.sqrt(sigmaB**2.0 + alpha**2.0*alldatasig)#sigmafactor*np.sqrt(abs(alldatasig))
        
        
        if sigtype == 'Ar17':
            results = self.linearandexponentialfit(x=self.x
                                                   ,sigtype=sigtype)
        else:
            results = self.linearmodelfit(x=self.x
                                          ,sigtype=sigtype)
        
        
        self.print_results(results)

        
        sigmawithplasma = self.plot_results(results
                                            ,alltimesig=alltimesig
                                            ,alldatasig=alldatasig
                                            ,alldatasigma=alldatasigma
                                            ,timelengthbeforeLBO=timelengthbeforeLBO
                                            ,numofbins=numofbins
                                            ,fsize=fsize
                                            ,linewidth=linewidth)
        
        
        middletime = ((results['time'][-1]-results['time'][0]) / 2.0) + results['time'][0]
        if sigtype == 'Ar17':
            intensityatmiddletime = self.linearandexponentialmodel(coeff=results['params'],x=[middletime])
            #sigmabackground = 0.0
        else:
            intensityatmiddletime = linearmodel(coeff=results['params'],x=[middletime])#middletime*results['params'][0]+results['params'][1]
        
        alpha = np.sqrt(abs(sigmawithplasma**2.0-sigmabackground**2.0) / intensityatmiddletime)
        
        print('sigmawithplasma: ',sigmawithplasma)
        print('sigmawithplasma**2.0: ',sigmawithplasma**2.0)
        print('sigmabackground: ',sigmabackground)
        print('sigmabackground**2.0: ',sigmabackground**2.0)
        print('(sigmawithplasma**2.0-sigmabackground**2.0): ',(sigmawithplasma**2.0-sigmabackground**2.0))
        
        print('middletime: ',middletime)
        print('intensityatmiddletime: ',intensityatmiddletime)
        print('alpha: ',alpha)
        
        
        print('All done')









































class linearbackgroundfitter:

    def linearmodel(self
                    ,coeff
                    ,x=None
                    ,numsig=None
                    ,lengthsig1=None
                    ,lengthsig2=None
                    ,lengthsig3=None):
                    #,residual_keywords=None):
        """
        Evaluate a linear function to the background of line radiation before 
        and after and LBO injection

        Fitting is done as a function of x 
        where x is the time window before LBO and after LBO
        Note that x window is the same for ALL signals (this might be changed later)
        """
        #x = self.x #residual_keywords['x']
        #numsig = 7 #residual_keywords['numsig']
        #lengthsig1 = residual_keywords['lengthsig1']
        #lengthsig2 = residual_keywords['lengthsig2']
        #lengthsig3 = residual_keywords['lengthsig3']
        y = np.zeros(len(x))

        for ii in range(lengthsig1):
            y[ii] = coeff[0]*x[ii] + coeff[1]
            
        #for ii in range(lengthsig1,(lengthsig2+lengthsig1)):
            #y[ii] = coeff[2]*x[ii] + coeff[3]
            
        for jj in range(numsig-2):
            #for ii in range((lengthsig2+lengthsig1)+lengthsig3*jj,lengthsig3*(jj+1)+(lengthsig2+lengthsig1)):
            for ii in range((lengthsig1)+lengthsig3*jj,lengthsig3*(jj+1)+(lengthsig1)):
                y[ii] = coeff[2*(jj+1)]*x[ii] + coeff[2*(jj+1)+1]
                

        return y

    def linearmodelresidual(self
                            ,coeff
                            ,x=None
                            ,numsig=None
                            ,lengthsig1=None
                            ,lengthsig2=None
                            ,lengthsig3=None):
                            #,residual_keywords=None):
        """
        Return the residual for the loglinearstrahl funciton.
        This method can be bassed to mir_mpfit.
        """

        yfit = self.linearmodel(coeff=coeff
                                ,x=x
                                ,numsig=numsig
                                ,lengthsig1=lengthsig1
                                ,lengthsig2=lengthsig2
                                ,lengthsig3=lengthsig3)
        
        residual = (self.data - yfit) / self.sigma
          
        return {'residual':residual}

    def linearmodelfit(self
                       ,x=None
                       ,numsig=None
                       ,lengthsig1=None
                       ,lengthsig2=None
                       ,lengthsig3=None):
                       #,residual_keywords=None):
        
        residual_keywords = {'x':x
                             ,'numsig':numsig
                             ,'lengthsig1':lengthsig1
                             ,'lengthsig2':lengthsig2
                             ,'lengthsig3':lengthsig3}
        
        
        coeffinitialguess = np.zeros(numsig*2)+100.0
        
        #coeffinitialguess=[ {[1.0,1.0]} for jj in range(len(coeffinitialguess))]
        

        parinfo=[{'step':0.0001,'fixed':True,'limited':[0,0],'limits':[0.0,0.0]} for jj in range(len(coeffinitialguess))]
        
        for ii in range(len(coeffinitialguess)):
            parinfo[ii]['fixed'] = False
            

        parinfoarray = np.zeros(len(parinfo))
        parinfoarrayfixedbool = []
        for ii in range(len(parinfo)):
            parinfoarray[ii] = parinfo[ii]['step']
            parinfoarrayfixedbool.append(parinfo[ii]['fixed'])

        mp = mpfit.mpfit(self.linearmodelresidual
                         ,coeffinitialguess
                         ,parinfo=parinfo
                         ,residual_keywords=residual_keywords)
        
        
        logging.info(mp.errmsg)
        logging.info('mpfit status:')
        logging.info(mp.statusString())
        
        if mp.perror is None: mp.perror = coeffinitialguess*0
            
        yfit = self.linearmodel(coeff=mp.params
                                ,x=x
                                ,numsig=numsig
                                ,lengthsig1=lengthsig1
                                ,lengthsig2=lengthsig2
                                ,lengthsig3=lengthsig3)
                                #,residual_keywords=residual_keywords)
        
        
        return {'description':'Linear: y[ii+(jj*len(x))] = coeff[0+(jj*2)] - coeff[1+(jj*2)]*x[ii]'
                ,'summedchisquared':mp.fnorm
                ,'time':x
                ,'numsig':numsig
                ,'lengthsig1':lengthsig1
                ,'lengthsig2':lengthsig2
                ,'lengthsig3':lengthsig3
                ,'fit':yfit
                ,'params':mp.params
                ,'perror':mp.perror
                ,'coeffinitial':coeffinitialguess
                ,'minparamstepsize': parinfoarray
                ,'fitstatus':mp.status
                ,'fititerations':mp.niter
                ,'numfunctcalls':mp.nfev
                ,'fixedbool': parinfoarrayfixedbool}

    def print_results(self
                      ,results):
                      #,residual_keywords=None):
        
        print('{} {}{}'.format('The fit status is:',results['fitstatus'],'\n'))
        print('{} {}{}'.format('The number of iterations completed:',results['fititerations'],'\n'))
        print('{} {}{}'.format('The number of calls to linearmodel:',results['numfunctcalls'],'\n'))
        print('Results:')
        print('{:>5s}  {:>6s}  {:>4s}  {:>15s}  {:>15s}  {:>6s}'.format(
            'param'
            ,'start value'
            ,'fit'
            ,'error'
            ,'minstepsize'
            ,'fixed'))
        
        for ii in range(len(results['params'])):
            print('{:>5d}  {:>10.4f}  {:>6.8f} +/- {:>4.8f}  {:>4.8f}  {:>6s}'.format(
                ii
                ,results['coeffinitial'][ii]
                ,results['params'][ii]
                ,results['perror'][ii]
                ,results['minparamstepsize'][ii]
                ,str(results['fixedbool'][ii])))
        print('\n','The total chisquared value is: ', results['summedchisquared'],'\n')
        
    def plot_results(self
                     ,results
                     ,alltimeAr17=None
                     ,alltimeqsx=None
                     ,alltimeHexos=None
                     ,alldataAr17=None
                     ,alldataqsx=None
                     ,alldataHexos=None):
        
        # Plot the fit results
        plt.close()

        fsize=(10.2,8)
        linewidth = 160
        startstr = 'Start: '+(np.array_str(results['coeffinitial'],max_line_width=linewidth))[1:-1]
        resultstr = 'Result: '+(np.array_str(results['params'],max_line_width=linewidth))[1:-1]
        perrorstr = 'Perror: '+(np.array_str(results['perror'],max_line_width=linewidth))[1:-1]

        title = startstr+'\n'+resultstr+'\n'+perrorstr

        len1 = results['lengthsig1']
        len2 = results['lengthsig2']
        len3 = results['lengthsig3']
        
        
        
        figAr17 = plt.figure(0)
        gsAr17 = gridspec.GridSpec(2,1,height_ratios=[2,1])
        ax1Ar17 = plt.subplot(gsAr17[0])
        #figAr17.suptitle(title,fontsize=15)
        #ax1Ar17.errorbar(self.x[:len1],self.data[:len1],yerr=self.sigma[:len1], fmt='bo', ms=2)
        ax1Ar17.errorbar(alltimeAr17,alldataAr17, fmt='bo', ms=2)
        ax1Ar17.scatter(self.x[:len1], results['fit'][:len1], color='red',s=6)
        ax1Ar17.plot(self.x[:len1], results['fit'][:len1], color='red')
        ax1Ar17.set_ylabel('Intensity (arb)',fontsize=18)
        #ax1Ar17.set_xlim(0,4.5)
        plt.setp(ax1Ar17.get_xticklabels(), visible=False)
    
        ax2Ar17 = plt.subplot(gsAr17[1],sharex=ax1Ar17)
        ax2Ar17.scatter(self.x[:len1],(1/self.sigma[:len1])*(self.data[:len1]-results['fit'][:len1]),s=6)
        ax2Ar17.set_ylabel('Weighted residual',fontsize=18)
        ax2Ar17.set_xlabel('Time (s)',fontsize=18)
        ax2Ar17.set_xlim(0,5.0)
        plt.subplots_adjust(hspace=.0)
        
        figAr17 = plt.figure(1)
        gsAr17 = gridspec.GridSpec(2,1,height_ratios=[2,1])
        ax1Ar17 = plt.subplot(gsAr17[0])
        #figAr17.suptitle(title,fontsize=15)
        #ax1Ar17.errorbar(self.x[:len1],self.data[:len1],yerr=self.sigma[:len1], fmt='bo', ms=2)
        ax1Ar17.errorbar(alltimeAr17,np.log(alldataAr17), fmt='bo', ms=2)
        ax1Ar17.scatter(self.x[:len1],np.log(results['fit'][:len1]), color='red',s=6)
        ax1Ar17.plot(self.x[:len1],np.log(results['fit'][:len1]), color='red')
        ax1Ar17.set_ylabel('Intensity (arb)',fontsize=18)
        ax1Ar17.set_yscale('log')
        ax1Ar17.set_ylim(7,11)
        #ax1Ar17.set_xlim(0,4.5)
        plt.setp(ax1Ar17.get_xticklabels(), visible=False)
    
        ax2Ar17 = plt.subplot(gsAr17[1],sharex=ax1Ar17)
        ax2Ar17.scatter(self.x[:len1],(1/self.sigma[:len1])*(self.data[:len1]-results['fit'][:len1]),s=6)
        ax2Ar17.set_ylabel('Weighted residual',fontsize=18)
        ax2Ar17.set_xlabel('Time (s)',fontsize=18)
        ax2Ar17.set_xlim(0,5.0)
        plt.subplots_adjust(hspace=.0)
        
        backgroundAr17 = np.zeros(len(alltimeAr17))
        for ii in range(len(alltimeAr17)):
            backgroundAr17[ii] = results['params'][0]*alltimeAr17[ii]+results['params'][1]
        
        fig0 = plt.figure()
        plt.scatter(alltimeAr17,alldataAr17, c='blue',s=8)
        plt.plot(alltimeAr17,alldataAr17, c='blue')
        plt.scatter(self.x[:len1], results['fit'][:len1], color='red',s=6)
        plt.plot(self.x[:len1], results['fit'][:len1], color='red')
        plt.xlabel('Time (s)',fontsize=18)
        plt.ylabel('Intensity (arb)',fontsize=18)
        plt.title('Spatially integrated Ar17: 171123036',fontsize=18)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlim(0,4.25)

        '''
        fig1 = plt.figure()
        plt.scatter(alltimeAr17,alldataAr17-backgroundAr17, c='blue',s=8)
        plt.xlabel('Time (s)',fontsize=18)
        plt.ylabel('Intensity (arb)',fontsize=18)
        '''
        
        fig2 = plt.figure()
        plt.scatter(alltimeAr17,alldataAr17-backgroundAr17, c='blue',s=8)
        plt.plot(alltimeAr17,alldataAr17-backgroundAr17, c='blue')
        plt.xlabel('Time (s)',fontsize=18)
        plt.ylabel('Intensity (arb)',fontsize=18)
        #plt.title('Spatially integrated Ar17: 171123036',fontsize=18)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlim(1.85,3.95)
        plt.ylim(-1000,50000)

        '''
        figqsx = plt.figure(3)
        gsqsx = gridspec.GridSpec(2,1,height_ratios=[2,1])
        ax1qsx = plt.subplot(gsqsx[0])
        #figqsx.suptitle(title,fontsize=15)
        #ax1qsx.errorbar(self.x[:len1],self.data[:len1],yerr=self.sigma[:len1], fmt='bo', ms=2)
        ax1qsx.errorbar(alltimeqsx,alldataqsx, fmt='bo', ms=2)
        ax1qsx.scatter(self.x[len1:len1+len2], results['fit'][len1:len1+len2], color='red',s=6)
        ax1qsx.plot(self.x[len1:len1+len2], results['fit'][len1:len1+len2], color='red')
        ax1qsx.set_ylabel('Intensity (arb)',fontsize=18)
        plt.setp(ax1qsx.get_xticklabels(), visible=False)
    
        ax2qsx = plt.subplot(gsqsx[1],sharex=ax1qsx)
        ax2qsx.scatter(self.x[len1:len1+len2],self.sigma[len1:len1+len2]*(self.data[len1:len1+len2]-results['fit'][len1:len1+len2]),s=6)
        ax2qsx.set_ylabel('Weighted residual',fontsize=18)
        ax2qsx.set_xlabel('Time (s)',fontsize=18)
        plt.subplots_adjust(hspace=.0)
        
        backgroundqsx = np.zeros(len(alltimeqsx))
        for ii in range(len(alltimeqsx)):
            backgroundqsx[ii] = results['params'][2]*alltimeqsx[ii]+results['params'][3]
   
        fig4 = plt.figure()
        plt.scatter(alltimeqsx,alldataqsx, c='blue',s=8)
        plt.plot(alltimeqsx,alldataqsx, c='blue')
        plt.scatter(self.x[len1:len1+len2], results['fit'][len1:len1+len2], color='red',s=6)
        plt.plot(self.x[len1:len1+len2], results['fit'][len1:len1+len2], color='red')
        plt.xlabel('Time (s)',fontsize=18)
        plt.ylabel('Intensity (arb)',fontsize=18)
        plt.title('Spatially integrated qsx: 171123036',fontsize=18)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlim(0,4.25)
        
        fig5 = plt.figure()
        plt.scatter(alltimeqsx,alldataqsx-backgroundqsx, c='blue',s=8)
        plt.plot(alltimeqsx,alldataqsx-backgroundqsx, c='blue')
        plt.xlabel('Time (s)',fontsize=18)
        plt.ylabel('Intensity (arb)',fontsize=18)
        #plt.title('Spatially integrated qsx: 171123036',fontsize=18)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlim(1.85,3.95)
        plt.ylim(-1000,50000)
        '''
        
        length = len1
        length2 = len(alltimeHexos)
        
        figFe6 = plt.figure(6,figsize=fsize)
        gsFe6 = gridspec.GridSpec(2,1,height_ratios=[2,1])
        ax1Fe6 = plt.subplot(gsFe6[0])
        figFe6.suptitle(title,fontsize=15)
        #ax1Fe6.errorbar(self.x[length:length+len3],self.data[length:length+len3],yerr=self.sigma[length:length+len3], fmt='bo', ms=2)
        ax1Fe6.errorbar(alltimeHexos,alldataHexos[:length2], fmt='bo', ms=2)
        ax1Fe6.scatter(self.x[length:length+len3], results['fit'][length:length+len3], color='red')
        ax1Fe6.plot(self.x[length:length+len3], results['fit'][length:length+len3], color='red')
        ax1Fe6.set_ylabel('Intensity (arb)',fontsize=15)
        #ax1Fe6.set_xlim(0,4.5)
        plt.setp(ax1Fe6.get_xticklabels(), visible=False)
        #ax1Fe6.set_xlim(2.3,2.4)
    
        ax2Fe6 = plt.subplot(gsFe6[1],sharex=ax1Fe6)
        ax2Fe6.scatter(self.x[length:length+len3],(1/self.sigma[length:length+len3])*(self.data[length:length+len3]-results['fit'][length:length+len3]))
        ax2Fe6.set_ylabel('Weighted residual',fontsize=15)
        ax2Fe6.set_xlabel('Time (s)',fontsize=15)
        ax2Fe6.set_xlim(0,5.0)
        plt.subplots_adjust(hspace=.0)
        #ax2Fe6.set_xlim(2.3,2.4)
        
        figFe6 = plt.figure(7,figsize=fsize)
        gsFe6 = gridspec.GridSpec(2,1,height_ratios=[2,1])
        ax1Fe6 = plt.subplot(gsFe6[0])
        figFe6.suptitle(title,fontsize=15)
        #ax1Fe6.errorbar(self.x[length:length+len3],self.data[length:length+len3],yerr=self.sigma[length:length+len3], fmt='bo', ms=2)
        ax1Fe6.errorbar(alltimeHexos,np.log(alldataHexos[:length2]), fmt='bo', ms=2)
        ax1Fe6.scatter(self.x[length:length+len3], np.log(results['fit'][length:length+len3]), color='red')
        ax1Fe6.plot(self.x[length:length+len3], np.log(results['fit'][length:length+len3]), color='red')
        ax1Fe6.set_yscale('log')
        ax1Fe6.set_ylim(7,10)
        #ax1Fe6.set_xlim(0,4.5)
        ax1Fe6.set_ylabel('Intensity (arb)',fontsize=15)
        plt.setp(ax1Fe6.get_xticklabels(), visible=False)
        #ax1Fe6.set_xlim(2.3,2.4)
    
        ax2Fe6 = plt.subplot(gsFe6[1],sharex=ax1Fe6)
        ax2Fe6.scatter(self.x[length:length+len3],(1/self.sigma[length:length+len3])*(self.data[length:length+len3]-results['fit'][length:length+len3]))
        ax2Fe6.set_ylabel('Weighted residual',fontsize=15)
        ax2Fe6.set_xlabel('Time (s)',fontsize=15)
        ax2Fe6.set_xlim(0,5.0)
        plt.subplots_adjust(hspace=.0)
        #ax2Fe6.set_xlim(2.3,2.4)
        
        length = length + len3
        
        figFe12 = plt.figure(12,figsize=fsize)
        gsFe12 = gridspec.GridSpec(2,1,height_ratios=[2,1])
        ax1Fe12 = plt.subplot(gsFe12[0])
        figFe12.suptitle(title,fontsize=15)
        #ax1Fe12.errorbar(self.x[length:length+len3],self.data[length:length+len3],yerr=self.sigma[length:length+len3], fmt='bo', ms=2)
        ax1Fe12.errorbar(alltimeHexos,alldataHexos[length2:length2*2], fmt='bo', ms=2)
        ax1Fe12.scatter(self.x[length:length+len3], results['fit'][length:length+len3], color='red')
        ax1Fe12.plot(self.x[length:length+len3], results['fit'][length:length+len3], color='red')
        ax1Fe12.set_ylabel('Intensity (arb)',fontsize=15)
        #ax1Fe12.set_xlim(0,4.5)
        plt.setp(ax1Fe12.get_xticklabels(), visible=False)
    
        ax2Fe12 = plt.subplot(gsFe12[1],sharex=ax1Fe12)
        ax2Fe12.scatter(self.x[length:length+len3],(1/self.sigma[length:length+len3])*(self.data[length:length+len3]-results['fit'][length:length+len3]))
        ax2Fe12.set_ylabel('Weighted residual',fontsize=15)
        ax2Fe12.set_xlabel('Time (s)',fontsize=15)
        ax2Fe12.set_xlim(0,5.0)
        plt.subplots_adjust(hspace=.0)
        
        
        length = length + len3
        
        figFe18 = plt.figure(18,figsize=fsize)
        gsFe18 = gridspec.GridSpec(2,1,height_ratios=[2,1])
        ax1Fe18 = plt.subplot(gsFe18[0])
        figFe18.suptitle(title,fontsize=15)
        #ax1Fe18.errorbar(self.x[length:length+len3],self.data[length:length+len3],yerr=self.sigma[length:length+len3], fmt='bo', ms=2)
        ax1Fe18.errorbar(alltimeHexos,alldataHexos[length2*2:length2*3], fmt='bo', ms=2)
        ax1Fe18.scatter(self.x[length:length+len3], results['fit'][length:length+len3], color='red')
        ax1Fe18.plot(self.x[length:length+len3], results['fit'][length:length+len3], color='red')
        ax1Fe18.set_ylabel('Intensity (arb)',fontsize=15)
        #ax1Fe18.set_xlim(0,4.5)
        plt.setp(ax1Fe18.get_xticklabels(), visible=False)
    
        ax2Fe18 = plt.subplot(gsFe18[1],sharex=ax1Fe18)
        ax2Fe18.scatter(self.x[length:length+len3],(1/self.sigma[length:length+len3])*(self.data[length:length+len3]-results['fit'][length:length+len3]))
        ax2Fe18.set_ylabel('Weighted residual',fontsize=15)
        ax2Fe18.set_xlabel('Time (s)',fontsize=15)
        ax2Fe18.set_xlim(0,5.0)
        plt.subplots_adjust(hspace=.0)
        
        
        length = length + len3
        
        figFe19 = plt.figure(19,figsize=fsize)
        gsFe19 = gridspec.GridSpec(2,1,height_ratios=[2,1])
        ax1Fe19 = plt.subplot(gsFe19[0])
        figFe19.suptitle(title,fontsize=15)
        #ax1Fe19.errorbar(self.x[length:length+len3],self.data[length:length+len3],yerr=self.sigma[length:length+len3], fmt='bo', ms=2)
        ax1Fe19.errorbar(alltimeHexos,alldataHexos[length2*3:length2*4], fmt='bo', ms=2)
        ax1Fe19.scatter(self.x[length:length+len3], results['fit'][length:length+len3], color='red')
        ax1Fe19.plot(self.x[length:length+len3], results['fit'][length:length+len3], color='red')
        ax1Fe19.set_ylabel('Intensity (arb)',fontsize=15)
        #ax1Fe19.set_xlim(0,4.5)
        plt.setp(ax1Fe19.get_xticklabels(), visible=False)
    
        ax2Fe19 = plt.subplot(gsFe19[1],sharex=ax1Fe19)
        ax2Fe19.scatter(self.x[length:length+len3],(1/self.sigma[length:length+len3])*(self.data[length:length+len3]-results['fit'][length:length+len3]))
        ax2Fe19.set_ylabel('Weighted residual',fontsize=15)
        ax2Fe19.set_xlabel('Time (s)',fontsize=15)
        ax2Fe19.set_xlim(0,5.0)
        plt.subplots_adjust(hspace=.0)
        
        
        length = length + len3
        
        figFe21 = plt.figure(21,figsize=fsize)
        gsFe21 = gridspec.GridSpec(2,1,height_ratios=[2,1])
        ax1Fe21 = plt.subplot(gsFe21[0])
        figFe21.suptitle(title,fontsize=15)
        #ax1Fe21.errorbar(self.x[length:length+len3],self.data[length:length+len3],yerr=self.sigma[length:length+len3], fmt='bo', ms=2)
        ax1Fe21.errorbar(alltimeHexos,alldataHexos[length2*4:length2*5], fmt='bo', ms=2)
        ax1Fe21.scatter(self.x[length:length+len3], results['fit'][length:length+len3], color='red')
        ax1Fe21.plot(self.x[length:length+len3], results['fit'][length:length+len3], color='red')
        ax1Fe21.set_ylabel('Intensity (arb)',fontsize=15)
        #ax1Fe21.set_xlim(0,4.5)
        plt.setp(ax1Fe21.get_xticklabels(), visible=False)
    
        ax2Fe21 = plt.subplot(gsFe21[1],sharex=ax1Fe21)
        ax2Fe21.scatter(self.x[length:length+len3],(1/self.sigma[length:length+len3])*(self.data[length:length+len3]-results['fit'][length:length+len3]))
        ax2Fe21.set_ylabel('Weighted residual',fontsize=15)
        ax2Fe21.set_xlabel('Time (s)',fontsize=15)
        ax2Fe21.set_xlim(0,5.0)
        plt.subplots_adjust(hspace=.0)
        
        
        length = length + len3
        
        figFe22 = plt.figure(22,figsize=fsize)
        gsFe22 = gridspec.GridSpec(2,1,height_ratios=[2,1])
        ax1Fe22 = plt.subplot(gsFe22[0])
        figFe22.suptitle(title,fontsize=15)
        #ax1Fe22.errorbar(self.x[length:length+len3],self.data[length:length+len3],yerr=self.sigma[length:length+len3], fmt='bo', ms=2)
        ax1Fe22.errorbar(alltimeHexos,alldataHexos[length2*5:length2*6], fmt='bo', ms=2)
        ax1Fe22.scatter(self.x[length:length+len3], results['fit'][length:length+len3], color='red')
        ax1Fe22.plot(self.x[length:length+len3], results['fit'][length:length+len3], color='red')
        ax1Fe22.set_ylabel('Intensity (arb)',fontsize=15)
        #ax1Fe22.set_xlim(0,4.5)
        plt.setp(ax1Fe22.get_xticklabels(), visible=False)
        #ax1Fe22.set_xlim(2.33,2.4)
    
        ax2Fe22 = plt.subplot(gsFe22[1],sharex=ax1Fe22)
        ax2Fe22.scatter(self.x[length:length+len3],(1/self.sigma[length:length+len3])*(self.data[length:length+len3]-results['fit'][length:length+len3]))
        ax2Fe22.set_ylabel('Weighted residual',fontsize=15)
        ax2Fe22.set_xlabel('Time (s)',fontsize=15)
        ax2Fe22.set_xlim(0,5.0)
        plt.subplots_adjust(hspace=.0)
        #ax2Fe22.set_xlim(2.33,2.4)
        
          

    def runlinearmodelfit(self
                          ,experimentID):
        
        
        #raw_data_access.getImages returns ---> {'image':image_array, 'time':time_array, 'expo_period':expo_period, 'expo_time':expo_time}
        dataAr17 = raw_data_access.getImages('ar17',experimentID,showimage=False)
        #dataqsx = raw_data_access.getImages('qsx',experimentID,showimage=False)
        
        
        time_arrayAr17=np.loadtxt('C:/Users/petr/Desktop/171123036ar17datatime.csv',delimiter=',')
        totalsumImarrayAr17=np.loadtxt('C:/Users/petr/Desktop/171123036ar17data.csv',delimiter=',')
        
        
        time_arrayAr17 = dataAr17['time']
        #time_arrayqsx = dataqsx['time']
        
        #ar17timesumImarray = np.sum(dataAr17['image'],axis=0)
        #horsumImarray = np.sum(dataAr17['image'],axis=1)
        vertsumImarrayAr17 = np.sum(dataAr17['image'],axis=2)
        totalsumImarrayAr17 = np.sum(vertsumImarrayAr17,axis=1)
        #vertsumImarrayqsx = np.sum(dataqsx['image'],axis=2)
        #totalsumImarrayqsx = np.sum(vertsumImarrayqsx,axis=1)
        
        timetostartbeforeLBOAr17 = 1.85
        
        timetostartbeforeLBO = 1.5
        timetoendbeforeLBO = 2.05
        timetostartafterLBO = 3.75
        timetoendafterLBO = 3.95
        
        indexstartbeforeLBOAr17 = (np.abs(time_arrayAr17 - timetostartbeforeLBOAr17)).argmin()
        indexendbeforeLBOAr17 = (np.abs(time_arrayAr17 - timetoendbeforeLBO)).argmin()
        indexstartafterLBOAr17 = (np.abs(time_arrayAr17 - timetostartafterLBO)).argmin()
        indexendafterLBOAr17 = (np.abs(time_arrayAr17 - timetoendafterLBO)).argmin()
        
        timewindowbeforeLBOAr17 = time_arrayAr17[indexstartbeforeLBOAr17:indexendbeforeLBOAr17]
        timewindowbackgroundAr17 = np.append(timewindowbeforeLBOAr17,time_arrayAr17[indexstartafterLBOAr17:indexendafterLBOAr17])
        backgrounddatafortimewindowAr17 = np.append(totalsumImarrayAr17[indexstartbeforeLBOAr17:indexendbeforeLBOAr17],totalsumImarrayAr17[indexstartafterLBOAr17:indexendafterLBOAr17])
        sigmaAr17 = np.sqrt(abs(backgrounddatafortimewindowAr17))

        '''
        indexstartbeforeLBOqsx = (np.abs(time_arrayqsx - timetostartbeforeLBOAr17)).argmin()
        indexendbeforeLBOqsx = (np.abs(time_arrayqsx - timetoendbeforeLBO)).argmin()
        indexstartafterLBOqsx = (np.abs(time_arrayqsx - timetostartafterLBO)).argmin()
        indexendafterLBOqsx = (np.abs(time_arrayqsx - timetoendafterLBO)).argmin()

        timewindowbeforeLBOqsx = time_arrayqsx[indexstartbeforeLBOqsx:indexendbeforeLBOqsx]
        timewindowbackgroundqsx = np.append(timewindowbeforeLBOqsx,time_arrayqsx[indexstartafterLBOqsx:indexendafterLBOqsx])
        backgrounddatafortimewindowqsx = np.append(totalsumImarrayqsx[indexstartbeforeLBOqsx:indexendbeforeLBOqsx],totalsumImarrayqsx[indexstartafterLBOqsx:indexendafterLBOqsx])
        sigmaqsx = np.sqrt(abs(backgrounddatafortimewindowqsx))
        '''
        
        #Begin Hexos data  
        totsignalHexos2 = raw_data_access.read_hexos_data(experimentID,2)
        wavelengthHexos2 = totsignalHexos2['wavelength'] #wavelength is stored LONG to SHORT
        shottimeHexos2 = totsignalHexos2['shottime']
        
        indexofFeVII = (np.abs(wavelengthHexos2 - 17.86)).argmin()
        indexofFeXIII = (np.abs(wavelengthHexos2 - 20.37)).argmin()
        indexofFeXIX = (np.abs(wavelengthHexos2 - 10.84)).argmin()
        indexofFeXX = (np.abs(wavelengthHexos2 - 12.18)).argmin()
        indexofFeXXII = (np.abs(wavelengthHexos2 - 11.72)).argmin()
        indexofFeXXIII = (np.abs(wavelengthHexos2 - 13.27)).argmin()
        
        halfwindow = 2
        
        signalFeVII = np.sum(totsignalHexos2['signal'][:,indexofFeVII-halfwindow:indexofFeVII+halfwindow+1],axis=1)
        signalFeXIII = np.sum(totsignalHexos2['signal'][:,indexofFeXIII-halfwindow:indexofFeXIII+halfwindow+1],axis=1)
        signalFeXIX = np.sum(totsignalHexos2['signal'][:,indexofFeXIX-halfwindow:indexofFeXIX+halfwindow+1],axis=1)
        signalFeXX = np.sum(totsignalHexos2['signal'][:,indexofFeXX-halfwindow:indexofFeXX+halfwindow+1],axis=1)
        signalFeXXII = np.sum(totsignalHexos2['signal'][:,indexofFeXXII-halfwindow:indexofFeXXII+halfwindow+1],axis=1)
        signalFeXXIII = np.sum(totsignalHexos2['signal'][:,indexofFeXXIII-halfwindow:indexofFeXXIII+halfwindow+1],axis=1)
        
        intensityshiftFeVII = np.sum(signalFeVII[-1000:]) / len(signalFeVII[-1000:])
        intensityshiftFeXIII = np.sum(signalFeXIII[-1000:]) / len(signalFeXIII[-1000:])
        intensityshiftFeXIX = np.sum(signalFeXIX[-1000:]) / len(signalFeXIX[-1000:])
        intensityshiftFeXX = np.sum(signalFeXX[-1000:]) / len(signalFeXX[-1000:])
        intensityshiftFeXXII = np.sum(signalFeXXII[-1000:]) / len(signalFeXXII[-1000:])
        intensityshiftFeXXIII = np.sum(signalFeXXIII[-1000:]) / len(signalFeXXIII[-1000:])
        
        signalFeVII = signalFeVII+abs(intensityshiftFeVII)
        signalFeXIII = signalFeXIII+abs(intensityshiftFeXIII)
        signalFeXIX = signalFeXIX+abs(intensityshiftFeXIX)
        signalFeXX = signalFeXX+abs(intensityshiftFeXX)
        signalFeXXII = signalFeXXII+abs(intensityshiftFeXXII)
        signalFeXXIII = signalFeXXIII+abs(intensityshiftFeXXIII)
        
        
        indexstartbeforeLBOHexos = (np.abs(shottimeHexos2 - timetostartbeforeLBO)).argmin()
        indexendbeforeLBOHexos = (np.abs(shottimeHexos2 - timetoendbeforeLBO)).argmin()
        indexstartafterLBOHexos = (np.abs(shottimeHexos2 - timetostartafterLBO)).argmin()
        indexendafterLBOHexos = (np.abs(shottimeHexos2 - timetoendafterLBO)).argmin()
        
        timewindowbeforeLBOHexos = shottimeHexos2[indexstartbeforeLBOHexos:indexendbeforeLBOHexos]
        timewindowbackgroundHexos = np.append(timewindowbeforeLBOHexos,shottimeHexos2[indexstartafterLBOHexos:indexendafterLBOHexos])
        
        backgrounddatafortimewindowFeVII = np.append(signalFeVII[indexstartbeforeLBOHexos:indexendbeforeLBOHexos],signalFeVII[indexstartafterLBOHexos:indexendafterLBOHexos])
        backgrounddatafortimewindowFeXIII = np.append(signalFeXIII[indexstartbeforeLBOHexos:indexendbeforeLBOHexos],signalFeXIII[indexstartafterLBOHexos:indexendafterLBOHexos])
        backgrounddatafortimewindowFeXIX = np.append(signalFeXIX[indexstartbeforeLBOHexos:indexendbeforeLBOHexos],signalFeXIX[indexstartafterLBOHexos:indexendafterLBOHexos])        
        backgrounddatafortimewindowFeXX = np.append(signalFeXX[indexstartbeforeLBOHexos:indexendbeforeLBOHexos],signalFeXX[indexstartafterLBOHexos:indexendafterLBOHexos])        
        backgrounddatafortimewindowFeXXII = np.append(signalFeXXII[indexstartbeforeLBOHexos:indexendbeforeLBOHexos],signalFeXXII[indexstartafterLBOHexos:indexendafterLBOHexos])        
        backgrounddatafortimewindowFeXXIII = np.append(signalFeXXIII[indexstartbeforeLBOHexos:indexendbeforeLBOHexos],signalFeXXIII[indexstartafterLBOHexos:indexendafterLBOHexos])        
        
        factor = 1.0
        
        sigmaFeVII = factor*np.sqrt(abs(backgrounddatafortimewindowFeVII))
        sigmaFeXIII = factor*np.sqrt(abs(backgrounddatafortimewindowFeXIII))
        sigmaFeXIX = factor*np.sqrt(abs(backgrounddatafortimewindowFeXIX))
        sigmaFeXX = factor*np.sqrt(abs(backgrounddatafortimewindowFeXX))
        sigmaFeXXII = factor*np.sqrt(abs(backgrounddatafortimewindowFeXXII))
        sigmaFeXXIII = factor*np.sqrt(abs(backgrounddatafortimewindowFeXXIII))

        alldataHexos = np.concatenate((signalFeVII
                                       ,signalFeXIII
                                       ,signalFeXIX
                                       ,signalFeXX
                                       ,signalFeXXII
                                       ,signalFeXXIII))

        self.x = np.concatenate((timewindowbackgroundAr17
                                #,timewindowbackgroundqsx
                                ,timewindowbackgroundHexos
                                ,timewindowbackgroundHexos
                                ,timewindowbackgroundHexos
                                ,timewindowbackgroundHexos
                                ,timewindowbackgroundHexos
                                ,timewindowbackgroundHexos))

        self.data = np.concatenate((backgrounddatafortimewindowAr17
                                   #,backgrounddatafortimewindowqsx
                                   ,backgrounddatafortimewindowFeVII
                                   ,backgrounddatafortimewindowFeXIII
                                   ,backgrounddatafortimewindowFeXIX
                                   ,backgrounddatafortimewindowFeXX
                                   ,backgrounddatafortimewindowFeXXII
                                   ,backgrounddatafortimewindowFeXXIII))
        
        self.sigma = np.concatenate((sigmaAr17
                                    #,sigmaqsx
                                    ,sigmaFeVII
                                    ,sigmaFeXIII
                                    ,sigmaFeXIX
                                    ,sigmaFeXX
                                    ,sigmaFeXXII
                                    ,sigmaFeXXIII))
        

        residual_keywords = {'x':self.x
                             ,'numsig':7
                             ,'lengthsig1':len(timewindowbackgroundAr17)
                             #,'lengthsig2':len(timewindowbackgroundqsx)
                             ,'lengthsig3':len(timewindowbackgroundHexos)}
   
        results = self.linearmodelfit(x=self.x
                                      ,numsig=7
                                      ,lengthsig1=len(timewindowbackgroundAr17)
                                      #,lengthsig2=len(timewindowbackgroundqsx)
                                      ,lengthsig3=len(timewindowbackgroundHexos))
                                      #,residual_keywords=residual_keywords)
        
        self.print_results(results)
                           #,residual_keywords=residual_keywords)
        
        self.plot_results(results
                          ,alltimeAr17=time_arrayAr17
                          #,alltimeqsx=time_arrayqsx
                          ,alltimeHexos=shottimeHexos2
                          ,alldataAr17=totalsumImarrayAr17
                          #,alldataqsx=totalsumImarrayqsx
                          ,alldataHexos=alldataHexos)
        
        lengthsig1 = len(timewindowbackgroundAr17)
        lengthsig3 = len(timewindowbackgroundHexos)
        weightedresidualAr17 = (1/sigmaAr17)*(backgrounddatafortimewindowAr17-results['fit'][:lengthsig1])
        weightedresidualFeVII = (1/sigmaFeVII)*(backgrounddatafortimewindowFeVII-results['fit'][lengthsig1:lengthsig1+lengthsig3])
        weightedresidualFeXIII = (1/sigmaFeXIII)*(backgrounddatafortimewindowFeXIII-results['fit'][lengthsig1+lengthsig3:lengthsig1+(2*lengthsig3)])
        weightedresidualFeXIX = (1/sigmaFeXIX)*(backgrounddatafortimewindowFeXIX-results['fit'][lengthsig1+(2*lengthsig3):lengthsig1+(3*lengthsig3)])
        weightedresidualFeXX = (1/sigmaFeXX)*(backgrounddatafortimewindowFeXX-results['fit'][lengthsig1+(3*lengthsig3):lengthsig1+(4*lengthsig3)])
        weightedresidualFeXXII = (1/sigmaFeXXII)*(backgrounddatafortimewindowFeXXII-results['fit'][lengthsig1+(4*lengthsig3):lengthsig1+(5*lengthsig3)])
        weightedresidualFFeXXIII = (1/sigmaFeXXIII)*(backgrounddatafortimewindowFeXXIII-results['fit'][lengthsig1+(5*lengthsig3):])

        weightedresidualAr17sorted = np.sort(weightedresidualAr17)
        weightedresidualFeVIIsorted = np.sort(weightedresidualFeVII)
        weightedresidualFeXIIIsorted = np.sort(weightedresidualFeXIII)
        weightedresidualFeXIXsorted = np.sort(weightedresidualFeXIX)
        weightedresidualFeXXsorted = np.sort(weightedresidualFeXX)
        weightedresidualFeXXIIsorted = np.sort(weightedresidualFeXXII)
        weightedresidualFeXXIIIsorted = np.sort(weightedresidualFFeXXIII)
        
        Ar17dotplotvaluearray = np.zeros(len(weightedresidualAr17sorted))-0.6
        Hexosdotplotvaluearray = np.zeros(len(weightedresidualFeVIIsorted))-0.6
        cumprobAr17 = np.zeros(len(weightedresidualAr17sorted))
        cumprobHexos = np.zeros(len(weightedresidualFeVIIsorted))
        
        for ii in range(len(weightedresidualAr17sorted)):
            cumprobAr17[ii] = ((ii+1) / (len(weightedresidualAr17sorted)+1))-0.5
            
        for ii in range(len(weightedresidualFeVIIsorted)):
            cumprobHexos[ii] = ((ii+1) / (len(weightedresidualFeVIIsorted)+1))-0.5
        
        #print('len(cumprobAr17) ',len(cumprobAr17))
        #print('len(weightedresidualAr17sorted) ',len(weightedresidualAr17sorted))
        #print('len(cumprobHexos) ',len(cumprobHexos))
        #print('len(weightedresidualFeVIIsorted) ',len(weightedresidualFeVIIsorted))
        plt.figure()
        plt.scatter(cumprobAr17,weightedresidualAr17sorted)
        plt.scatter(Ar17dotplotvaluearray,weightedresidualAr17sorted)
        
        plt.figure()
        plt.hist(weightedresidualAr17sorted,bins=50)
        
        plt.figure()
        plt.scatter(cumprobHexos,weightedresidualFeVIIsorted)
        plt.scatter(Hexosdotplotvaluearray,weightedresidualFeVIIsorted)

        plt.figure()
        plt.hist(weightedresidualFeVIIsorted,bins=50)

        plt.figure()
        plt.scatter(cumprobHexos,weightedresidualFeXIIIsorted)
        plt.scatter(Hexosdotplotvaluearray,weightedresidualFeXIIIsorted)

        plt.figure()
        plt.hist(weightedresidualFeXIIIsorted,bins=50)

        plt.figure()
        plt.scatter(cumprobHexos,weightedresidualFeXIXsorted)
        plt.scatter(Hexosdotplotvaluearray,weightedresidualFeXIXsorted)

        plt.figure()
        plt.hist(weightedresidualFeXIXsorted,bins=50)

        plt.figure()
        plt.scatter(cumprobHexos,weightedresidualFeXXsorted)
        plt.scatter(Hexosdotplotvaluearray,weightedresidualFeXXsorted)

        plt.figure()
        plt.hist(weightedresidualFeXXsorted,bins=50)

        plt.figure()
        plt.scatter(cumprobHexos,weightedresidualFeXXIIsorted)
        plt.scatter(Hexosdotplotvaluearray,weightedresidualFeXXIIsorted)

        plt.figure()
        plt.hist(weightedresidualFeXXIIsorted,bins=50)
        
        plt.figure()
        plt.scatter(cumprobHexos,weightedresidualFeXXIIIsorted)
        plt.scatter(Hexosdotplotvaluearray,weightedresidualFeXXIIIsorted)

        plt.figure()
        plt.hist(weightedresidualFeXXIIIsorted,bins=50)
       
        print('All done')
