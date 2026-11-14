"""Public, read-only Project Nyx dashboard; no source URL or write key."""
from datetime import datetime, timezone
import logging
import os
import pandas as pd
import requests
import streamlit as st
from storage import draws,predictions
from model import walkforward, ALPHABET
from prizes import evaluate_prize, benchmark_odds

st.set_page_config(page_title='Project Nyx',page_icon='🌙',layout='wide')
st.html(
    '<script data-goatcounter="https://darkdragon.goatcounter.com/count" '
    'async src="https://gc.zgo.at/count.js"></script>',
    unsafe_allow_javascript=True,
)
st.title('🌙 Project Nyx')
st.caption('Statistical sequence lab · exploratory forecasts, not reliable winning-number predictions')

@st.cache_data(ttl=10,show_spinner=False)
def goatcounter_total():
    token=os.getenv('GOATCOUNTER_API_KEY','')
    if not token:
        try:
            token=str(st.secrets.get('GOATCOUNTER_API_KEY',''))
        except (FileNotFoundError,KeyError):
            pass
    if not token:
        raise RuntimeError('Missing private configuration: GOATCOUNTER_API_KEY')

    response=requests.get(
        'https://darkdragon.goatcounter.com/api/v0/stats/total',
        params={'start':'1970-01-01T00:00:00Z'},
        headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'},
        timeout=10,
    )
    response.raise_for_status()
    total=response.json()['total']
    if not isinstance(total,int) or isinstance(total,bool):
        raise ValueError('GoatCounter returned an invalid total')
    return total

@st.fragment(run_every='15s')
def render_visitor_count():
    st.markdown('---')
    try:
        total=goatcounter_total()
        checked_at=datetime.now(timezone.utc).strftime('%H:%M:%S UTC')
        st.success(f'GoatCounter connected · {total:,} all-time total · checked {checked_at}')
    except RuntimeError as error:
        logging.warning('GoatCounter unavailable: %s',error)
        st.warning(f'GoatCounter unavailable: {error}')
    except requests.RequestException as error:
        status=error.response.status_code if error.response is not None else 'no HTTP response'
        logging.warning('GoatCounter request failed (%s; %s)',type(error).__name__,status)
        st.error(f'GoatCounter request failed ({status}). Check the app logs.')
    except (KeyError,TypeError,ValueError):
        logging.warning('GoatCounter returned an invalid response.')
        st.error('GoatCounter returned an invalid response. Check the app logs.')

@st.cache_data(ttl=600)
def load():return draws(),predictions()

try:
    history,guesses=load()
except Exception:
    st.error('The data service is temporarily unavailable. Check dashboard secrets and database policies.')
    render_visitor_count()
    st.stop()
if not history:
    st.info('No data yet. Run the GitHub collection workflow after configuring its secrets.')
    render_visitor_count()
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
    prize=evaluate_prize(p['predicted_letter'],p['predicted_digits'],
                         observed['letter'],observed['digits'])
    records.append({'draw':p['draw_number'],'forecast':p['predicted_letter']+p['predicted_digits'],
                    'actual':observed['letter']+observed['digits'],
                    'digit_position_hits':hits,'letter_hit':p['predicted_letter']==observed['letter'],
                    'exact_hit':p['predicted_letter']==observed['letter'] and p['predicted_digits']==observed['digits'],
                    'model':p['model_name'],'valid_training_order':valid,
                    'prize_tier':prize.tier,'prize_rs_reference':prize.amount,
                    'first_consecutive':prize.prefix,'last_consecutive':prize.suffix})
if records:
    scored=pd.DataFrame(records)
    # Invalid training order means a row could be a hindsight prediction; never count it.
    eligible=scored[scored['valid_training_order']].copy()
    if len(eligible)!=len(scored):
        st.warning('Some stored forecasts are invalid for prospective evaluation; excluded from the summary.')
    st.metric('Exact matches (valid saved forecasts)',f'{int(eligible["exact_hit"].sum())}/{len(eligible)}')
    st.metric('Prize-winning forecasts (reference tiers)',
              f'{int((eligible["prize_rs_reference"]>0).sum())}/{len(eligible)}')
    st.metric('Reference prize total (not verified winnings)',
              f'Rs. {int(eligible["prize_rs_reference"].sum()):,}')
    st.metric('Mean correct digit positions',f'{eligible["digit_position_hits"].mean():.2f} / 6' if len(eligible) else 'N/A',help='Uniform-random expectation is 0.6 out of 6')
    st.caption('Amounts use the 2024 gazette reference; jackpot varies by draw. This evaluation counts the largest applicable tier, and does not prove a ticket was purchased or paid.')
    st.dataframe(scored.sort_values('draw',ascending=False),use_container_width=True,hide_index=True)
else:st.info('No saved forecast has been matched against an actual draw yet.')

st.subheader('Prize-aware random baseline')
odds=benchmark_odds(len(ALPHABET))
a,b,c=st.columns(3)
a.metric('Super prize: chance',f'1 in {round(1/odds["super_exact_probability"]):,}')
b.metric('6 numbers: chance',f'1 in {round(1/odds["six_digits_probability"]):,}')
c.metric('Any listed prize: chance',f'{odds["any_prize_probability"]:.2%}')
st.caption('Assumes 25 equally likely letters, six independent uniform digits, and at most one payable prize tier. The any-prize baseline includes letter, first digit, or last digit. Prize-aware scoring measures outcomes, not an increased ability to predict independent draws.')

st.subheader('Nyx 2.0 · statistical model comparison')
if len(history)>20:
    selected_classical,scores=walkforward(history)
    model_df=pd.DataFrame([{
        'model':k,
        'validation_log_loss':v['validation_mean_log_loss'],
        'holdout_log_loss':v['test_mean_log_loss'],
        'holdout_brier':v['test_mean_brier_score'],
        'holdout_symbol_hits':v['test_mean_symbol_hits'],
        'holdout_cases':v['test_cases'],
        'selected_classical':k==selected_classical,
    } for k,v in scores.items()])
    st.dataframe(model_df,use_container_width=True,hide_index=True)
    st.caption('Classical models are selected using validation log loss. The later holdout is reporting-only. Lower log loss and Brier score are better; small differences do not establish a repeatable lottery pattern.')
else:
    st.info('More draw history is needed for walk-forward testing.')

st.subheader('Nyx 2.0 · experimental deep learning')
latest_saved=max(guesses,key=lambda p:int(p['draw_number'])) if guesses else None
latest_meta=(latest_saved or {}).get('probabilities') or {}
deep_report=latest_meta.get('deep_learning') or {}
if deep_report.get('available'):
    st.caption('LSTM and GRU are trained by the private scheduled runner. The dashboard reads their saved evaluation; it does not retrain models.')
    selected_overall=latest_meta.get('selected_model',latest_saved.get('model_name','unknown'))
    st.metric('Model selected for latest forecast',selected_overall.upper())
    deep_rows=[]
    for model_name,model_score in deep_report.get('scores',{}).items():
        validation=model_score.get('validation') or {}
        holdout=model_score.get('test') or {}
        deep_rows.append({
            'model':model_name.upper(),
            'validation_log_loss':validation.get('mean_log_loss'),
            'holdout_log_loss':holdout.get('mean_log_loss'),
            'holdout_brier':holdout.get('mean_brier_score'),
            'holdout_symbol_hits':holdout.get('mean_symbol_hits'),
            'holdout_prize_hits':holdout.get('prize_hit_count'),
            'mean_reference_prize_rs':holdout.get('mean_reference_prize_rs'),
            'holdout_cases':holdout.get('cases'),
        })
    if deep_rows:
        st.dataframe(pd.DataFrame(deep_rows),use_container_width=True,hide_index=True)
    st.caption(f"Lookback: {deep_report.get('lookback_draws', 'N/A')} draws · Validation cases: {deep_report.get('validation_cases', 'N/A')} · Untouched holdout cases: {deep_report.get('test_cases', 'N/A')}. Model choice uses validation log loss, never holdout results.")
else:
    reason=deep_report.get('reason') or 'The next scheduled private run will assess whether enough history and the optional PyTorch dependency are available.'
    st.info(reason)

st.caption('Important: LSTM/GRU are experiments, not proven prediction advantages. Draw results may be independent and random. Prize-hit counts and reference prize values are historical evaluation statistics, not expected future winnings.')
render_visitor_count()
