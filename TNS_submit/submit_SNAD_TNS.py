#!/usr/bin/env python
"""
Send SNAD objects to TNS for reporting
-Patrick Aleo (Github: patrickaleo) adapted from Chien-Hsiu Lee <chien-hsiu.lee@noirlab.edu> 20230706 (yyyymmdd)

"""
import astropy.time as atime
from collections import OrderedDict
import time

import sys
import os
import string
import getpass
import requests
from requests.auth import HTTPBasicAuth
from bs4 import BeautifulSoup
import json
from datetime import datetime
from sys import platform as sys_pf

#import PySimpleGUI as sg

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import rc, rcParams
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import pandas as pd
import numpy as np
from astropy.io import ascii, fits
from astropy import wcs
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import SkyCoord, SkyOffsetFrame
from astropy.coordinates import concatenate as SkyConcat
from astropy.time import Time
import astropy.units as u
import astropy.table as at
from astropy.table import Table
#import reproject
import warnings
warnings.filterwarnings("ignore")

import argparse
parser = argparse.ArgumentParser(description="usage: python3 submit_SNAD_TNS.py -subtype [args] -private_key [args] -reporter [args] -plot_first_detection [args] -refmag [args] -err_refmag [args] -snad_name [args] -tns_type [args] -ztf_id [args] -use_common_remark [args] -oid [args]\n\n e.g. python3 submit_SNAD_TNS.py -subtype TNS -private_key [REDACTED] -reporter 'Patrick Aleo (UIUC)' -plot_first_detection True -refmag 20.943 -err_refmag 0.057 -snad_name SNAD180 -tns_type PSN -ztf_id ZTF18abnwdwh -use_common_remark Miner -oid 825114400003486")
parser.add_argument("-subtype", "--submissiontype", dest = "subtype", help="Submission Type (TNS or Sandbox)", type=str, required=True)
parser.add_argument("-private_key", "--private_key", dest = "private_key", help="Private API Key for SNAD_bot.", type=str, required=True)
parser.add_argument("-reporter", "--reporter", dest = "reporter", help="Reporter Name on AT Report. Example: 'Patrick Aleo (UIUC)'. NOTE: Needs to be in single quotes if you have spaces!", type=str, required=True)
parser.add_argument("-plot_first_detection", "--plot_first_detection", dest = "plot_first_detection", help="If to plot first detection (True or False)", type=str, required=True)
#parser.add_argument("-approx_detection_mjd", "--approx_detection_mjd", dest = "approx_detection_mjd", help="Tells algorithm to search around this MJD for first point above 3 sigma from reference mag to use as first detection.", type=str, required=True)
parser.add_argument("-refmag", "--refmag", dest = "refmag", help="ZTF reference magnitude", type=str, required=True)
parser.add_argument("-err_refmag", "--err_refmag", dest = "err_refmag", help="ZTF reference magnitude error", type=str, required=True)
parser.add_argument("-snad_name", "--snad_name", dest = "snad_name", help="Internal SNAD Catalog name", type=str, required=True)
parser.add_argument("-tns_type", "--tns_type", dest = "tns_type", help="Indicate valid TNS Type (PSN, PNV, AGN, NUC, Other)", type=str, required=True)
parser.add_argument("-ztf_id", "--ztf_id", dest = "ztf_id", help="ZTF ID. If none, write 'None'", type=str, required=True)
parser.add_argument("-use_common_remark", "--use_common_remark", dest = "use_common_remark", help="Use common remark for TNS entry (AAD, Miner, PineForest, or Custom). If Custom, you will be prompted to enter a custom remark.", type=str, required=True)
parser.add_argument("-oid", "--oid", nargs='+', dest = "oid", help="ZTF OID", type=str, required=True)
parser.add_argument("-upload_plot", "--upload_plot", dest="upload_plot", help="If True, upload PNG light curve before submission", type=str, required=False, default="False")

args = parser.parse_args()

print("Submission Type: {}, SNAD ZTF OID: {}".format(args.subtype, args.oid))

"""
Interfacing SNAD with Transient Name Server:

For more detail on programmatic access of TNS via API,
please see the help page of TNS (https://www.wis-tns.org/content/tns-getting-started).

Transient Name Server (TNS) is the official IAU
mechanism for reporting new astronomical transients (ATs).
We can report filtered ANTARES locus of certain type or candidate objects with the TNS API as follows.

The first step is to import TNS API.
This is based on the sample python code by Nikola Knezevic in https://www.wis-tns.org/sites/default/files/api/tns_api_bulk_report.py.zip .
Please also create a directory json_reports_for_sending/ under the current path.
The reports will be stored under this directory.

In this notebook it will first test the report against TNS sandbox (TNS="sandbox.wis-tns.org").
For offical reports, please switch the url to (TNS="www.wis-tns.org").
The "api_key" is for SNAD_bot; please keep it confidential.
"""


def flux_error(mag, err_mag):
    err_down = 10**(-0.4*(mag - err_mag)) - 10**(-0.4*mag)
    err_up = 10**(-0.4*mag) - 10**(-0.4*(mag + err_mag))
    err_flux = 0.5*(err_down + err_up)
    return err_flux

def mag_error(flux, err_flux):
    err_down = abs(-2.5*np.log10(flux + err_flux) + 2.5*np.log10(flux))
    err_up = abs(-2.5*np.log10(flux) + 2.5*np.log10(flux - err_flux))
    err_down = np.where(np.isnan(err_down), 1e4, err_down)
    err_up = np.where(np.isnan(err_up), 1e4, err_up)
    return err_down, err_up

def SendTNS():

    ###########################################################################################
    ####################################### PARAMETERS ########################################
    reporter_name = str(args.reporter) # Name of reporter to go on TNS report
    print(f"Reporter name: {reporter_name}")

    upload_plot = args.upload_plot.lower() == "true" # upload lightcurve png or not
    
    if args.subtype == 'TNS':
        TNS="www.wis-tns.org"
    elif args.subtype == 'Sandbox':
        TNS="sandbox.wis-tns.org"
    else:
        print('Choose TNS or Sandbox for -subtype flag!')
    url_tns_api="https://"+TNS+"/api"

    YOUR_BOT_ID="87994"
    YOUR_BOT_NAME="SNAD_bot"
    api_key=args.private_key

    list_of_filenames="Here put your list of filenames for uploading."
    report_filename="Here put your report filename."
    # report type can only be "tsv" or "json"
    report_type="Here put the type of your report."
    id_report="Here put your report ID for getting report's reply."

    # current working directory
    cwd=os.getcwd()
    # folder containing files for uploading
    upload_folder=os.path.join(cwd,'files_for_uploading')
    # folder containing tsv reports for sending
    tsv_reports_folder=os.path.join(cwd,'tsv_reports_for_sending')
    # folder containing json reports for sending
    json_reports_folder=os.path.join(cwd,'json_reports_for_sending')

    # http errors
    http_errors={
    304: 'Error 304: Not Modified: There was no new data to return.',
    400: 'Error 400: Bad Request: The request was invalid. '\
         'An accompanying error message will explain why.',
    403: 'Error 403: Forbidden: The request is understood, but it has '\
         'been refused. An accompanying error message will explain why.',
    404: 'Error 404: Not Found: The URI requested is invalid or the '\
         'resource requested, such as a category, does not exists.',
    500: 'Error 500: Internal Server Error: Something is broken.',
    503: 'Error 503: Service Unavailable.'
    }

    # how many second to sleep
    SLEEP_SEC=5
    # max number of time to check response
    LOOP_COUNTER=60
    # keeping sys.stdout
    old_stdout=sys.stdout

    ###########################################################################################
    ###########################################################################################


    ###########################################################################################
    ######################################## FUNCTIONS ########################################

    # function for changing data to json format
    def format_to_json(source):
        # change data to json format and return
        parsed=json.loads(source,object_pairs_hook=OrderedDict)
        result=json.dumps(parsed,indent=4)
        return result

    # function for uploading files trough api
    def upload_files(url,list_of_files):
        try:
            # url for uploading files
            upload_url=url+'/set/file-upload'
            # headers
            headers={'User-Agent':'tns_marker{"tns_id":'+str(YOUR_BOT_ID)+', "type":"bot",'\
                     ' "name":"'+YOUR_BOT_NAME+'"}'}
            # api key data
            api_data={'api_key':api_key}
            # construct a dictionary of files and their data
            files_data={}
            for i in range(len(list_of_files)):
                file_name=list_of_files[i]
                file_path=os.path.join(upload_folder,file_name)
                key='files['+str(i)+']'
                if file_name.lower().endswith(('.asci', '.ascii')):
                    value=(file_name, open(file_path), 'text/plain')
                else:
                    value=(file_name, open(file_path,'rb'), 'application/fits')
                files_data[key]=value
            # upload all files using request module
            response=requests.post(upload_url, headers=headers, data=api_data, files=files_data)
            # return response
            return response
        except Exception as e:
            return [None,'Error message : \n'+str(e)]

    # function for sending tsv reports (AT or Classification)
    def send_tsv_report(url,tsv_report):
        try:
            # url for sending tsv reports
            tsv_url=url+'/csv-report'
            # headers
            headers={'User-Agent':'tns_marker{"tns_id":'+str(YOUR_BOT_ID)+', "type":"bot",'\
                     ' "name":"'+YOUR_BOT_NAME+'"}'}
            # api key data
            api_data={'api_key':api_key}
            # tsv report file path
            tsv_file_path=os.path.join(tsv_reports_folder,tsv_report)
            # read tsv data from file
            tsv_read=(tsv_report, open(tsv_file_path,'rb'))
            # construct a dictionary of tsv data
            tsv_data={'csv':tsv_read}
            # send tsv report using request module
            response=requests.post(tsv_url, headers=headers, data=api_data, files=tsv_data)
            # return response
            return response
        except Exception as e:
            return [None,'Error message : \n'+str(e)]

    # function for sending json reports (AT or Classification)
    def send_json_report(url,json_report):
        try:
            # url for sending json reports
            json_url=url+'/set/bulk-report'
            # headers
            headers={'User-Agent':'tns_marker{"tns_id":'+str(YOUR_BOT_ID)+', "type":"bot",'\
                     ' "name":"'+YOUR_BOT_NAME+'"}'}
            # json report file path
            json_file_path=os.path.join(json_reports_folder,json_report)
            # read json data from file
            json_read=format_to_json(open(json_file_path).read())
            # construct a dictionary of api key data and json data
            json_data={'api_key':api_key, 'data':json_read}
            # send json report using request module
            response=requests.post(json_url, headers=headers, data=json_data)
            # return response
            return response
        except Exception as e:
            return [None,'Error message : \n'+str(e)]

    # function for getting reply from report
    def reply(url, report_id):
        try:
            # url for getting report reply
            reply_url=url+'/get/bulk-report-reply'
            # headers
            headers={'User-Agent':'tns_marker{"tns_id":'+str(YOUR_BOT_ID)+', "type":"bot",'\
                     ' "name":"'+YOUR_BOT_NAME+'"}'}
            # construct a dictionary of api key data and report id
            reply_data={'api_key':api_key, 'report_id':report_id}
            # send report ID using request module
            response=requests.post(reply_url, headers=headers, data=reply_data)
            #print(response.status_code)
            #print(response.text)
            return response
        except Exception as e:
            return [None,'Error message : \n'+str(e)]

    # function that checks response and
    # returns True if everything went OK
    # or returns False if something went wrong
    def check_response(response):
        # if response exists
        if None not in response:
            # take status code of that response
            status_code=int(response.status_code)
            if status_code==200:
                # response as json data
                json_data=response.json()
                # id code
                id_code=str(json_data['id_code'])
                # id message
                id_message=str(json_data['id_message'])
                # print id code and id message
                print ("ID code = "+id_code)
                print ("ID message = "+id_message)
                # check if id code is 200 and id message OK
                if (id_code=="200" and id_message=="OK"):
                    return True
                #special case
                elif (id_code=="400" and id_message=="Bad request"):
                    return None
                else:
                    return False
            else:
                # if status code is not 200, check if it exists in
                # http errors
                if status_code in list(http_errors.keys()):
                    print (list(http_errors.values())
                           [list(http_errors.keys()).index(status_code)])
                else:
                    print ("Undocumented error.")
                return False
        else:
            # response doesn't exists, print error
            print (response[1])
            return False

    # find all occurrences of a specified key in json data
    # and return all values for that key
    def find_keys(key, json_data):
        if isinstance(json_data, list):
            for i in json_data:
                for x in find_keys(key, i):
                    yield x
        elif isinstance(json_data, dict):
            if key in json_data:
                yield json_data[key]
            for j in list(json_data.values()):
                for x in find_keys(key, j):
                    yield x

    # print feedback
    def print_feedback(json_feedback):
        # find all message id-s in feedback
        message_id=list(find_keys('message_id',json_feedback))
        # find all messages in feedback
        message=list(find_keys('message',json_feedback))
        # find all obj names in feedback
        objname=list(find_keys('objname',json_feedback))
        # find all new obj types in feedback
        new_object_type=list(find_keys('new_object_type',json_feedback))
        # find all new obj names in feedback
        new_object_name=list(find_keys('new_object_name',json_feedback))
        # find all new redshifts in feedback
        new_redshift=list(find_keys('new_redshift',json_feedback))
        # index counters for objname, new_object_type, new_object_name
        # and new_redshift lists
        n_o=0
        n_not=0
        n_non=0
        n_nr=0
        # messages to print
        msg=[]
        # go trough every message and print
        for j in range(len(message)):
            m=str(message[j])
            m_id=str(message_id[j])
            if m_id not in ['102','103','110']:
                if m.endswith('.')==False:
                    m=m+'.'
                if m_id=='100' or  m_id=='101':
                    m="Message = "+m+" Object name = "+str(objname[n_o])
                    global TNSobjname
                    TNSobjname = objname[n_o]
                    n_o=n_o+1
                elif m_id=='120':
                    m="Message = "+m+" New object type = "+str(new_object_type[n_not])
                    n_not=n_not+1
                elif m_id=='121':
                    m="Message = "+m+" New object name = "+str(new_object_name[n_non])
                    n_non=n_non+1
                elif m_id=='122' or  m_id=='123':
                    m="Message = "+m+" New redshift = "+str(new_redshift[n_nr])
                    n_nr=n_nr+1
                else:
                    m="Message = "+m
                msg.append(["Message ID = "+m_id,m])
        # return messages
        return msg

    # sending report id to get reply of the report
    # and printing that reply
    def print_reply(url,report_id):
        # sending reply using report id and checking response
        print ("Sending reply for the report id "+report_id+" ...")
        print(f"Sleeping for {SLEEP_SEC} seconds...")
        time.sleep(SLEEP_SEC)
        reply_res=reply(url, report_id)
        counter = SLEEP_SEC
        reply_res_check=check_response(reply_res)
        # if reply is sent
        if reply_res_check==True:
            print ("The report was successfully processed on the TNS.\n")
            # reply response as json data
            json_data=reply_res.json()
            # feedback of the response
            feedback=list(find_keys('feedback',json_data))
            # check if feedback is dict or list
            if type(feedback[0])==type([]):
                feedback=feedback[0]
            # go trough feedback
            for i in range(len(feedback)):
                # feedback as json data
                json_f=feedback[i]
                # feedback keys
                feedback_keys=list(json_f.keys())
                # messages for printing
                msg=[]
                # go trough feedback keys
                for j in range(len(feedback_keys)):
                    key=feedback_keys[j]
                    json_feed=json_f[key]
                    msg=msg+print_feedback(json_feed)
                if msg!=[]:
                    print ("-----------------------------------"\
                           "-----------------------------------" )
                    for k in range(len(msg)):
                        print (msg[k][0])
                        print (msg[k][1])
                    print ("-----------------------------------"\
                           "-----------------------------------\n")
        else:
            if (reply_res_check!=None):
                print ("The report doesn't exist on the TNS.")
            else:
                print ("The report was not processed on the TNS "\
                       "because of the bad request(s).")

    # Disable print
    def blockPrint():
        sys.stdout = open(os.devnull, 'w')

    # Restore print
    def enablePrint():
        sys.stdout.close()
        sys.stdout = old_stdout

    # sending tsv or json report (at or class) and printing reply
    def send_report(url, report, type_of_report):
        # sending report and checking response
        print ("Sending "+report+" to the TNS...")
        # choose which function to call
        if type_of_report=="tsv":
            response=send_tsv_report(url,report)
        else:
            response=send_json_report(url,report)
        response_check=check_response(response)
        # if report is sent
        if response_check==True:
            print ("The report was sent to the TNS.")
            # report response as json data
            json_data=response.json()
            # taking report id
            report_id=str(json_data['data']['report_id'])
            print ("Report ID = "+report_id)
            print ("")
            # sending report id to get reply of the report
            # and printing that reply
            # waiting for report to arrive before sending reply
            # for report id
            blockPrint()
            counter = 0
            while True:
                time.sleep(SLEEP_SEC)
                reply_response=reply(url,report_id)
                reply_res_check=check_response(reply_response)
                if reply_res_check!=False or counter >= LOOP_COUNTER:
                    break
                counter += 1
            enablePrint()
            print_reply(url,report_id)
        else:
            print ("The report was not sent to the TNS.")

    # uploading files and printing reply
    def upload(url, list_of_files):
        # upload files and checking response
        print ("Uploading files on the TNS...")
        response=upload_files(url,list_of_files)
        response_check=check_response(response)
        # if files are uploaded
        if response_check==True:
            print ("The following files are uploaded on the TNS : ")
            # response as json data
            json_data=response.json()
            # list of uploaded files
            uploaded_files=json_data['data']
            for i in range(len(uploaded_files)):
                print ("filename : "+str(uploaded_files[i]))
        else:
            print ("Files are not uploaded on the TNS.")
        print ("\n")

###########################################################################################
###########################################################################################

#Official Bulk Report

###########################################################################################
###########################################################################################

    if args.plot_first_detection == 'True':
        plot_status=True
    elif args.plot_first_detection == 'False':
        plot_status=False
    else:
        print('Choose True or False for -plot_first_detection flag!')
        sys.exit()

    refmag = float(args.refmag)
    err_refmag = float(args.err_refmag)
    xlim_l,xlim_u = 58178.0, 60125.0 # private DR23 time span # Change if first detection is not good
    ylim_bright,ylim_faint = 18.0, 23.0  # 18.0, 23.0 # Change if first detection is not good

    # if object has a ZTF alert ID, use that as the internal name, and save SNAD internal name to REMARKS.
    # if object does not have a ZTF alert ID, use SNAD internal name instead, and write in remarks there is no ZTF alert ID.
    internal_ztf_name = str(args.ztf_id)
    internal_snad_name = str(args.snad_name)
    tns_type = str(args.tns_type)

    if internal_ztf_name != 'None': # HAS ZTF Alert ID!
        print("Has ZTF Alert ID! Adding internal SNAD Object ID to the Remarks.")
        has_ztf_alert_id = True
        report_internal_name = internal_ztf_name # added to report later
    else: # No ZTF alert ID
        print("No ZTF Alert ID! Using internal SNAD Object ID instead.")
        has_ztf_alert_id = False
        report_internal_name = internal_snad_name # added to report later


    if has_ztf_alert_id:
        ZTF_ALERT_ID_REMARK = f"The ZTF Alert ID for this object is {internal_ztf_name}. The internal SNAD Object ID is {internal_snad_name}."
    else:
        ZTF_ALERT_ID_REMARK = f"There is no ZTF Alert ID for this object (i.e., not in the alert stream), so we report the internal SNAD Object ID instead. The internal SNAD Object ID is {internal_snad_name}."

    # Select the report type.
    if tns_type=='PSN':
        at_type="1"
    elif tns_type=='PNV':
        at_type="2"
    elif tns_type=='AGN':
        at_type="3"
    elif tns_type=='NUC':
        at_type="4"
    elif tns_type=='Other':
        at_type="0"
    else:
        print('Choose PSN, PNV, AGN, NUC, Other for -tns_type flag!')
        sys.exit()

    oid = list(args.oid) #sys.argv[24:]
    print('Number of ZTF OIDs:', len(oid))
    print('Submitting the following ZTF OID for TNS report on', datetime.now(), ':', oid, '\n')

    for obj_id in oid:

        # Send one total TNS report.
        # Use ZTF alert ID as the internal name when available, and add the SNAD internal name to the Remarks section.
        # If there is no ZTF alert ID to use as the internal name, we can use SNAD internal name (repeat in Remarks).

        # Common remark
        if args.use_common_remark=='AAD':
            REMARK = f"Transient identified by AAD (https://arxiv.org/pdf/1909.13260.pdf). See more info at: https://ztf.snad.space/view/{obj_id}. {ZTF_ALERT_ID_REMARK}"
        elif args.use_common_remark=='Miner':
            REMARK = f"Transient identified by SNAD Miner (https://arxiv.org/pdf/2111.11555.pdf). See more info at: https://ztf.snad.space/view/{obj_id}. {ZTF_ALERT_ID_REMARK}"
        elif args.use_common_remark=='PineForest':
            REMARK = f"Transient identified by Pine Forest. See more info at: https://ztf.snad.space/view/{obj_id}. {ZTF_ALERT_ID_REMARK}"
        elif args.use_common_remark=='Custom':
            # Taking input from the user
            REMARK = input("You selected to write a custom remark. What would you like to say?...")
            print(f"\n Recorded the following custom remark: \n{REMARK}\n! Moving on...")
        else:
            print('Write AAD, Miner, PineForest, or Custom for -use_common_remark flag!')
            sys.exit()


        # snad object:
        url = 'https://ztf.snad.space/latest/csv/' + f'{obj_id}'
        data = Table.from_pandas(pd.read_csv(url))
        xlim = [xlim_l,xlim_u]
        ylim = [ylim_bright,ylim_faint]


        # from ztf refmag or the last photometrical point for this OID
        ref = round(float(refmag), 2)
        err_ref = round(float(err_refmag), 2)

        #Load ra, dec
        page = requests.get(f"http://db.ztf.snad.space/api/v3/data/latest/oid/full/json?oid={obj_id}")

        soup = page.json()
        ra = round(soup[obj_id]['meta']['coord']['ra'], 5)
        dec = round(soup[obj_id]['meta']['coord']['dec'], 5)

        data = data[(data['mjd'] >= xlim_l) & (data['mjd'] <= xlim_u)]
        #print(data['mjd'])

        data['flux'] = 10**(-0.4*data['mag']) - 10**(-0.4*ref)
        data['fluxerr'] = np.sqrt(flux_error(data['mag'],data['magerr'])**2 + flux_error(ref,err_ref)**2)
        data['magerr_down'] = mag_error(data['flux'], data['fluxerr'])[0]
        data['magerr_up'] = mag_error(data['flux'], data['fluxerr'])[1]

        found_first_detection = False
        for i in data:
            if i['flux'] > 3*i['fluxerr']:
                tr_time = i['mjd']
                flux = i['flux']
                mag = round(-2.5*np.log10(i['flux']), 2)
                magerr_down = round(i['magerr_down'], 2)
                magerr_up = round(i['magerr_up'], 2)
                passband = i['filter']
                found_first_detection = True
                break

        if not found_first_detection:
            print(f"[WARNING] No detection above 3-sigma for OID {obj_id}. Skipping...")
            continue

        t_mjd = Time(tr_time, format='mjd', scale='utc')
        if plot_status==True:
            fig = plt.figure(figsize=(8,3))
            fig.subplots_adjust(left=.08, bottom=.15, right=.98, top=0.95)

            gs = GridSpec(1, 2, height_ratios=[1])
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])

            ax1.errorbar(x=data['mjd'],y=data['flux'],yerr=data['fluxerr'], marker='s', ms=2, mew=1, ls='none', zorder=-1)
            ax1.axvline(x=tr_time, color='red', ls='--', zorder=0)
            ax1.plot(tr_time, flux, 'ro', label='1st detection')
            ax1.set_xlabel('mjd')
            ax1.set_ylabel('relative flux')
            ax1.legend()

            ax2.errorbar(x=data['mjd'],y=-2.5*np.log10(data['flux']),yerr=(mag_error(data['flux'], data['fluxerr'])[0],mag_error(data['flux'], data['fluxerr'])[1]), marker='s', ms=2, mew=1, ls='none', zorder=-1)
            ax2.axvline(x=tr_time, color='red', ls='--', zorder=0)
            ax2.plot(tr_time, mag, 'ro', label='1st detection')
            ax2.axhline(y=ref, color='black', ls='--', zorder=1)
            ax2.text(x=data['mjd'].mean(), y=ref, s='reference', color='black', ha='center', va='bottom', fontsize=10, zorder=2)

            ax2.set_xlim(xlim)
            ax2.set_ylim(ylim)
            ax2.invert_yaxis()
            ax2.set_xlabel('mjd')
            ax2.set_ylabel('magnitude')
            ax2.legend()

            #t_mjd = Time(tr_time, format='mjd', scale='utc')
            print('First detection: t_UTC = %s, mag = %2.2f, magerr_down = %2.2f, magerr_up = %2.2f, filter = %s' %(t_mjd.iso, mag, magerr_down, magerr_up, passband))
            plt.show()

            # Taking input from the user
            string = input("Is this first detection ok? If yes, type 'y'. If no, type 'n' and change arguments in script...")

            # Output
            if string == 'y':
                print("Continuing on!...")
                pass
            elif string == 'n':
                print("Exiting script")
                break
            else:
                raise("type 'y' or 'n'")

        if not args.ztf_id=='None': # HAS ZTF Alert ID!
            ZTF_ID = str(args.ztf_id)
            PHOT_REMARK = f"For the first photometric point reference magnitude is {str(ref)} mag. ZTF DR OID {obj_id} ({passband}). ZTF Alert ID {args.ztf_id}."
        else: # NO ZTF Alert ID!
            ZTF_ID = np.nan
            PHOT_REMARK = f"For the first photometric point reference magnitude is {str(ref)} mag. ZTF DR OID {obj_id} ({passband}). No ZTF Alert ID found."

        #sorting out filter of the lastest detection
        # fid = 1 is ZTF-g and fid=2 is ZTF-r
        if passband == 'zg':
            filter_id="110"
        if passband == 'zr':
            filter_id="111"

        snad_data={
          "at_report": {
            "0": {
              "ra": {
                "value": str(ra),
        #        "error": "0.5",
                "units": "deg"
              },
              "dec": {
                "value": str(dec),
        #        "error": "0.5",
                "units": "deg"
              },
              "reporting_group_id": "95", #SNAD - 95; ANTARES - 115
              "discovery_data_source_id": "48", #ZTF
              "reporter": f"{reporter_name} on behalf of the SNAD team", #MAKE REPORTER NAME ARG
              "discovery_datetime":str(round(tr_time+2400000.5, 7)),
              "at_type": at_type, #1-PSN, 2-PNV...
              "host_name": "",
              "host_redshift": "",
              "transient_redshift": "",
              "internal_name": str(report_internal_name),
              "internal_name_format": {
                "prefix": "", #prefixStr
                "year_format": "", #YY
                "postfix": "" #postfixStr
              },
              "remarks": REMARK,
              "proprietary_period_groups": [
                "95"
              ],
              "proprietary_period": {
                "proprietary_period_value": "0",
                "proprietary_period_units": "days"
              },
              "non_detection": {
                "obsdate": "",
                "limiting_flux":"",
                "flux_units": "",
                "filter_value": "",
                "instrument_value": "",
                "exptime": "",
                "observer": "",
                "comments": "",
                "archiveid": "2", #1 - SDSS; 2 - DSS (all sky)
                "archival_remarks": ""
              },
              "photometry": {
                "photometry_group": {
                  "0": {
                    "obsdate":str(round(tr_time+2400000.5, 7)),
                    "flux":str(mag),
                    "flux_error":"",
                    "limiting_flux":"",
                    "flux_units": "1", #ABMag
                    "filter_value": filter_id,
                    "instrument_value": "196", #P48 Palomar 1.2m Oschin
                    "exptime": "",
                    "observer": "ZTF",
                    "comments": PHOT_REMARK #MAKE PHOTREMARK
                  }
                }
              }
            }
          }
        }

        # upload png BEFORE submit
        related_file_name = None  # default value
        if upload_plot:
            print("Uploading PNG plot before submission...")
            plot_name = f"{obj_id}.png"
            plot_path = os.path.join("OID_plots_to_add", plot_name)
            if os.path.exists(plot_path):
                headers = {
                    "User-Agent": f'tns_marker{{"tns_id":"{YOUR_BOT_ID}", "type":"bot", "name":"{YOUR_BOT_NAME}"}}'
                }
                upload_url = f"https://{TNS}/api/set/file-upload"
                files = {"files[0]": open(plot_path, "rb")}
                data_upload = {"api_key": api_key}

                resp = requests.post(upload_url, headers=headers, files=files, data=data_upload)

                if resp.status_code == 200:
                    #print(resp.json())
                    saved_plot_name = resp.json()["data"][0]
                    print(f"Uploaded PNG. Saved plot name: {saved_plot_name}")
                    related_file_name = saved_plot_name
                else:
                    print(f"[WARNING] Failed to upload PNG: {resp.status_code}, {resp.text}")
            else:
                print(f"[WARNING] PNG file does not exist: {plot_path}")

        if related_file_name:
            snad_data["at_report"]["0"]["related_files"] = {
                                                            "0": {
                                                                    "related_file_name": related_file_name,
                                                                    "related_file_comments": "Light curve plot."
                                                                }
                                                            }

        
        
        #TNS entry
        filename=obj_id+'_data.txt'

        if not os.path.isdir('./json_reports_for_sending'):
            os.makedirs('./json_reports_for_sending', exist_ok=True)

        with open('json_reports_for_sending/'+filename, 'w') as outfile:
            json.dump(snad_data, outfile)

        # send AT report
        report_filename=filename
        report_type="json"

        send_report(url_tns_api,report_filename,report_type)

        print("\nNow updating TNS entry to SNAD catalog...")
        #SNAD Catalog Master entry
        catalog_entry_fn=obj_id+'_catalog_entry.csv'

        if not os.path.isdir('./new_entry_for_SNAD_catalog_master'):
            os.makedirs('./new_entry_for_SNAD_catalog_master', exist_ok=True)

        try:
            catalog_entry_df = pd.DataFrame([[f'{str(internal_snad_name)}', ra, dec, obj_id, str(t_mjd.iso), f'{str(mag)}', magerr_down, magerr_up, ref, err_ref, 'AT '+f'{str(TNSobjname)}', tns_type, ZTF_ID, np.nan, np.nan, np.nan]], columns=['Name', 'R.A.', 'Dec.', 'OID', 'Discovery date (UT)', 'mag', 'er_down', 'er_up', 'ref', 'er_ref', 'TNS', 'Type', 'Comments', 'z_ph', 'er_z_ph', 'Remarks'])
        except:
            print("TNS name delayed. Check TNS name and add manually to both sheets (or redownload)!")
            catalog_entry_df = pd.DataFrame([[f'{str(internal_snad_name)}', ra, dec, obj_id, str(t_mjd.iso),
                                              f'{str(mag)}', magerr_down, magerr_up, ref, err_ref,
                                              'AT XXX', tns_type, ZTF_ID, np.nan, np.nan, np.nan]],
                                            columns=['Name', 'R.A.', 'Dec.', 'OID', 'Discovery date (UT)', 'mag',
                                                     'er_down', 'er_up', 'ref', 'er_ref', 'TNS', 'Type', 'Comments',
                                                     'z_ph', 'er_z_ph', 'Remarks'])
        catalog_entry_df.set_index('Name', inplace=True)
        catalog_entry_df.to_csv('./new_entry_for_SNAD_catalog_master/'+catalog_entry_fn, index=True)

        print(f"\nSaved to new entry to {catalog_entry_fn}! Now will update full catalog...")

        # Google sheet of master SNAD catalog

        sheet_url='https://docs.google.com/spreadsheets/d/1yLjNWffw2vUrzOpGx3yHwWs7lewbGuiQqgfFmuMC7ww/edit#gid=0'
        csv_export_url = sheet_url.replace('/edit#gid=', '/export?format=csv&gid=')
        snad_catalog_df = pd.read_csv(csv_export_url, index_col=0)
        catalog_entry_df = pd.read_csv('./new_entry_for_SNAD_catalog_master/'+catalog_entry_fn, index_col=0)
        updated_snad_catalog_df = pd.concat([snad_catalog_df, catalog_entry_df])
        updated_snad_catalog_df.to_csv('SNAD_catalog_master_test.csv', index=True)

        print("\nSaved to full SNAD catalog at SNAD_catalog_master_test.csv")
        print("Don't forget to reupload sheet before next TNS submission :) ")

        #Now reupload SNAD_catalog_master_test.csv to Google Sheets
        # GO to file --> import --> replace current sheet !!

        # Or just copy and paste from SNAD_catalog_master_test.csv honestly



###########################################################################################
###########################################################################################

def main():
    tns = SendTNS()
    return

if __name__ == "__main__":
    main()
