-- Run in Supabase SQL Editor once. Public dashboard gets SELECT only.
create table if not exists public.draws (
 draw_number bigint primary key check (draw_number > 0),
 draw_date date not null,
 letter text not null check (letter ~ '^[A-Z]$'),
 digits text not null check (digits ~ '^[0-9]{6}$'),
 inserted_at timestamptz not null default now()
);
create table if not exists public.predictions (
 draw_number bigint primary key check (draw_number > 0),
 predicted_letter text not null check (predicted_letter ~ '^[A-Z]$'),
 predicted_digits text not null check (predicted_digits ~ '^[0-9]{6}$'),
 model_name text not null,
 training_max_draw bigint not null check (training_max_draw < draw_number),
 created_at timestamptz not null default now(),
 probabilities jsonb not null default '{}'::jsonb
);

alter table public.draws enable row level security;
alter table public.predictions enable row level security;
-- This grants public VIEWING of draw/prediction records. The repository stays public too.
revoke all on public.draws from anon, authenticated;
revoke all on public.predictions from anon, authenticated;
grant select on public.draws to anon;
grant select on public.predictions to anon;

drop policy if exists nyx_draws_public_read on public.draws;
create policy nyx_draws_public_read on public.draws for select to anon using (true);
drop policy if exists nyx_predictions_public_read on public.predictions;
create policy nyx_predictions_public_read on public.predictions for select to anon using (true);

-- Stronger than RLS: immutable history and forecasts, including service-role writes.
create or replace function public.nyx_prevent_change() returns trigger
language plpgsql set search_path = '' as $$
begin
 raise exception 'Immutable Nyx records may not be updated or deleted';
end;$$;
drop trigger if exists nyx_draw_immutable on public.draws;
create trigger nyx_draw_immutable before update or delete on public.draws
for each row execute function public.nyx_prevent_change();
drop trigger if exists nyx_prediction_immutable on public.predictions;
create trigger nyx_prediction_immutable before update or delete on public.predictions
for each row execute function public.nyx_prevent_change();
