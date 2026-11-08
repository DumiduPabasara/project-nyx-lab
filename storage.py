"""REST API with separate least-privilege credentials for reading and writing."""
import os
import requests


def config(name):
    value = os.getenv(name, '')
    if value:
        return value
    if name in ('SUPABASE_URL','SUPABASE_ANON_KEY'):
        try:
            import streamlit as st
            return str(st.secrets.get(name, ''))
        except (ImportError, FileNotFoundError, KeyError):
            pass
    raise RuntimeError('Missing private configuration: ' + name)


def api(method, table, params=None, data=None, private=False):
    if table not in ('draws','predictions'):
        raise ValueError('Invalid table')
    if method != 'GET' and not private:
        raise ValueError('Writes are allowed only from the private job')
    key = config('SUPABASE_SERVICE_ROLE_KEY' if private else 'SUPABASE_ANON_KEY')
    url = config('SUPABASE_URL').rstrip('/')
    if not url.startswith('https://'):
        raise RuntimeError('SUPABASE_URL must use HTTPS')
    headers={'apikey':key,'Authorization':'Bearer '+key,
             'Content-Type':'application/json',
             'Prefer':'return=minimal' if method!='GET' else 'return=representation'}
    res=requests.request(method, url+'/rest/v1/'+table, params=params, json=data, headers=headers, timeout=30)
    # Avoid printing request URLs or credentials in exception traces.
    if res.status_code >= 400:
        raise RuntimeError(f'Database {table} {method} failed (HTTP {res.status_code})')
    return res.json() if res.content else []


def fetch_all(table, private=False):
    output=[]
    for offset in range(0,1000000,1000):
        batch=api('GET',table,{'select':'*','order':'draw_number.asc','limit':'1000','offset':str(offset)},private=private)
        output+=batch
        if len(batch)<1000: break
    return output


def draws(private=False):return fetch_all('draws',private)
def predictions(private=False):return fetch_all('predictions',private)


def insert_draw(item):
    existing=api('GET','draws',{'select':'draw_number,draw_date,letter,digits','draw_number':'eq.'+str(item['draw_number'])},private=True)
    if existing:
        for name in ('draw_date','letter','digits'):
            if str(existing[0][name])!=str(item[name]):
                raise ValueError(f'Conflicting existing result for draw {item["draw_number"]}')
        return False
    try:
        api('POST','draws',data=item,private=True)
        return True
    except RuntimeError:
        # A retry might race with another job. Confirm final state before accepting.
        existing=api('GET','draws',{'select':'draw_number,draw_date,letter,digits','draw_number':'eq.'+str(item['draw_number'])},private=True)
        if existing and all(str(existing[0][k])==str(item[k]) for k in ('draw_date','letter','digits')):
            return False
        raise


def insert_prediction(item):
    if api('GET','predictions',{'select':'draw_number','draw_number':'eq.'+str(item['draw_number'])},private=True):
        return False
    try:
        api('POST','predictions',data=item,private=True)
        return True
    except RuntimeError:
        if api('GET','predictions',{'select':'draw_number','draw_number':'eq.'+str(item['draw_number'])},private=True):
            return False
        raise
