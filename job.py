"""Cloud runner: import history, collect, evaluate, make one immutable future forecast."""
import argparse
import csv
import hashlib
import os
import random
from datetime import date,datetime,timedelta,timezone
from pathlib import Path
from model import ALPHABET,probability_vectors,walkforward
from deep_models import evaluate_deep_models, predict_next
from storage import draws,insert_draw,insert_prediction,predictions


def import_csv(path):
    count=0
    with Path(path).open(encoding='utf-8-sig',newline='') as file:
        for row in csv.DictReader(file):
            raw=(row.get('draw_date') or row.get('date') or '').strip()
            try: dt=date.fromisoformat(raw).isoformat()
            except ValueError: dt=datetime.strptime(raw.replace(',',''),'%A %B %d %Y').date().isoformat()
            digits=str(row.get('digits') or row.get('winning_number') or '').zfill(6)
            letter=(row.get('letter') or '').strip().upper()
            if len(digits)!=6 or not digits.isdigit() or letter not in ALPHABET:
                raise ValueError('Invalid history row')
            count+=int(insert_draw(dict(draw_number=int(row['draw_number']),draw_date=dt,letter=letter,digits=digits)))
    return count


def forecast(history):
    target=history[-1]['draw_number']+1
    if any(p['draw_number']==target for p in predictions(private=True)):
        return f'Draw {target}: already saved'
    classical_name,scores=walkforward(history)
    name=classical_name
    deep_report=evaluate_deep_models(history)
    if deep_report.get('available'):
        deep_name=deep_report['selected_model']
        classical_loss=scores[classical_name]['validation_mean_log_loss']
        deep_loss=deep_report['scores'][deep_name]['validation']['mean_log_loss']
        # Deep learning is promoted only when it wins on the same untouched
        # validation window; the later test window is reporting-only.
        if deep_loss is not None and classical_loss is not None and deep_loss < classical_loss:
            name=deep_name

    try:
        vectors=predict_next(history,name) if name in ('lstm','gru','ensemble') else probability_vectors(history,name)
    except Exception as exc:
        # Preserve the established forecast path if optional neural fitting fails.
        print(f'Neural forecast unavailable ({type(exc).__name__}); using classical model')
        name=classical_name
        vectors=probability_vectors(history,name)

    digest=hashlib.sha256(f'{target}/{history[-1]["letter"]}/{history[-1]["digits"]}'.encode()).hexdigest()
    rng=random.Random(int(digest,16))
    picks=[rng.choice([i for i,x in enumerate(v) if abs(x-max(v))<1e-12]) for v in vectors]
    record=dict(draw_number=target,predicted_letter=ALPHABET[picks[0]],
        predicted_digits=''.join(map(str,picks[1:])),model_name=name,
        training_max_draw=history[-1]['draw_number'],
        created_at=datetime.now(timezone.utc).isoformat(),
        probabilities={'vectors':vectors,'walkforward':scores,
                       'deep_learning':deep_report,
                       'selected_model':name,
                       'selection_rule':'lowest validation log loss; holdout test is reporting-only'})
    return f'Draw {target}: '+('saved' if insert_prediction(record) else 'already saved')+' ('+name+')'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--import-csv',help='Import LOCAL PRIVATE history CSV; never commit it')
    parser.add_argument('--skip-scrape',action='store_true')
    parser.add_argument('--max-pages',type=int,default=25)
    args=parser.parse_args()
    if not 0<=args.max_pages<=200:parser.error('max-pages must be 0..200')
    if args.import_csv:print('Imported:',import_csv(args.import_csv))
    items=draws(private=True)
    if not args.skip_scrape and args.max_pages:
        start=items[-1]['draw_number']+1 if items else int(os.environ['NYX_START_DRAW'])
        from collector import collect
        print('Collection:',collect(start,args.max_pages))
    items=draws(private=True)
    if not items:
        print('No results available yet; no prediction generated');return
    latest=date.fromisoformat(items[-1]['draw_date'])
    # A future forecast must be saved before the future draw takes place.
    # When backfilling old history, do not misrepresent guesses as prospective.
    if latest < datetime.now(timezone(timedelta(hours=5,minutes=30))).date()-timedelta(days=2):
        print('History is stale/backfilling; forecast deferred')
    else:
        print('Forecast:',forecast(items))
    print('Draw count:',len(items),'Prediction count:',len(predictions(private=True)))

if __name__=='__main__': main()
