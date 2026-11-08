"""Public, read-only Project Nyx dashboard; no source URL or write key."""
from datetime import datetime, timezone
import pandas as pd
import streamlit as st
from storage import draws,predictions
from model import walkforward

st.set_page_config(page_title='Project Nyx',page_icon='🌙',layout='wide')
st.title('🌙 Project Nyx')
st.caption('Statistical sequence lab · exploratory forecasts, not reliable winning-number predictions')

@st.cache_data(ttl=600)
def load():return draws(),predictions()

try:
    history,guesses=load()
except Exception:
    st.error('The data service is temporarily unavailable. Check dashboard secrets and database policies.')
    st.stop()
if not history:
    st.info('No data yet. Run the GitHub collection workflow after configuring its secrets.')
    st.stop()
by_draw={int(d['draw_number']):d for d in history}
latest=history[-1]
result=latest['letter']+' '+latest['digits']
active=next((p for p in guesses if int(p['draw_number'])>int(latest['draw_number'])),None)
a,b,c=st.columns(3)
a.metric('Latest draw',str(latest['draw_number']))
b.metric('Latest result',result)
c.metric('Stored predictions',len(guesses))
st.subheader('Next saved forecast')
if active:
    st.metric('Target draw '+str(active['draw_number']),active['predicted_letter']+' '+active['predicted_digits'])
    st.caption('Stored before that draw is known; no guarantee of predictability')
else:st.info('No pending forecast yet. The next scheduled job will generate one after data catches up.')

st.subheader('Historical records')
df=pd.DataFrame(history)
st.dataframe(df[['draw_number','draw_date','letter','digits']].sort_values('draw_number',ascending=False),use_container_width=True,hide_index=True)

st.subheader('Digit frequencies by position')
position=st.selectbox('Digit position',options=list(range(1,7)),format_func=lambda i:f'Position {i}')
frequency=pd.Series([r['digits'][position-1] for r in history]).value_counts().reindex(list('0123456789'),fill_value=0)
st.bar_chart(frequency.rename('Count'))

st.subheader('Prospective forecast evaluation')
records=[]
for p in guesses:
    observed=by_draw.get(int(p['draw_number']))
    if not observed:continue
    # Strictly verify this was generated with history preceding the target draw.
    valid=int(p['training_max_draw'])<int(p['draw_number'])
    hits=sum(x==y for x,y in zip(p['predicted_digits'],observed['digits']))
    records.append({'draw':p['draw_number'],'forecast':p['predicted_letter']+p['predicted_digits'],
                    'actual':observed['letter']+observed['digits'],
                    'digit_position_hits':hits,'letter_hit':p['predicted_letter']==observed['letter'],
                    'exact_hit':p['predicted_letter']==observed['letter'] and p['predicted_digits']==observed['digits'],
                    'model':p['model_name'],'valid_training_order':valid})
if records:
    scored=pd.DataFrame(records)
    st.metric('Exact matches (saved forecasts)',f'{int(scored["exact_hit"].sum())}/{len(scored)}')
    st.metric('Mean correct digit positions',f'{scored["digit_position_hits"].mean():.2f} / 6',help='Uniform-random expectation is 0.6 out of 6')
    st.dataframe(scored.sort_values('draw',ascending=False),use_container_width=True,hide_index=True)
else:st.info('No saved forecast has been matched against an actual draw yet.')

st.subheader('Walk-forward model comparison')
if len(history)>20:
    _,scores=walkforward(history)
    model_df=pd.DataFrame([dict(model=k,**v) for k,v in scores.items()])
    st.dataframe(model_df,use_container_width=True,hide_index=True)
    st.caption('Lower mean log loss is better. Comparisons use earlier draws only; exploratory comparisons are not proof of a predictive pattern.')
else:st.info('More draw history is needed for walk-forward testing.')
