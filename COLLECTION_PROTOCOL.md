# Pakistani face data: collection and consent protocol

Why this exists: the statement requires us to collect our own diverse data and to explain "data collection and consent process" in the presentation. No public Pakistani identity-labelled face set exists, so this is the main original contribution.

## Consent (every participant, before any photo)
Participant (or guardian if under 18; we do not enrol minors unless a guardian signs) reads and signs:

> I agree to let [Team name] take photographs of my face for a university competition project (Nemo's Deep Dive, NED University) on face recognition for Pakistani faces.
> - Photos are stored only on the team's devices / private cloud storage, labelled with a random ID (P001...), never my name.
> - They are used only to fine-tune and test a face recognition model. They will not be sold, published, or posted online.
> - I may withdraw at any time before **[date]** and my photos will be deleted.
> - Optional: I agree that my sex, age range and region may be recorded for fairness analysis.
>
> Signature / Date / Contact (kept separately from photos, deleted after the event)

Keep the signed forms (photo of paper is fine) in a folder separate from the images. The only link between a person and their ID is a private sheet that is deleted after the competition.

## Who to include (target 100+ people, minimum 40)
Spread across: gender, age (<18 excluded; 18-25, 26-40, 41-60, 60+), regions (Punjab, Sindh, KP, Balochistan, Gilgit-Baltistan/AJK, Muhajir/urban Karachi, Hazara/Pashtun etc.), skin tones, glasses, facial hair, hijab/dupatta/turban/cap.
Record the counts per group; thin groups are reported honestly in the fairness section rather than hidden.

## Photos per person (aim 12-20)
| tag in filename | what |
|---|---|
| `std` | frontal, good light (x3, slight pose changes) |
| `lowlight` | dim indoor / evening |
| `harsh` | direct sunlight / strong backlight |
| `cctv` | taken from 2-4 m, phone held high, cropped later; or screen-recorded low-res |
| `blur` | slight motion blur |
| `mask` | surgical mask |
| `cover` | hijab / dupatta / turban / cap (only if the person normally wears one) |
| `glasses` | with glasses (if worn) |
| `side` | ~30-45 degree yaw |
Use at least 2 different phones. Take photos on different days if possible (same-day only photos make verification look easier than it is).

## Folder layout
```
data/pk/raw/P001/P001_std_1.jpg
data/pk/raw/P001/P001_cctv_1.jpg
data/pk/people.csv    # person_id,gender,age_bracket,region,covering,glasses,facial_hair
```
Then `python ingest.py` builds `data/pk/manifest.csv` with an **identity-disjoint** train/val/test split (e.g. 60/15/25% of people).

## Rules that keep the evaluation honest
1. Splits are by person, never by photo. `evaluate.py` aborts if one person is in two splits.
2. Thresholds are chosen on val only. Test is touched once per final model.
3. Never train on test persons, public-dataset test sets, or the TechFest benchmark if shared.
4. Delete the person-to-ID sheet and signed forms after the event; keep the anonymised images only as long as the consent says.
