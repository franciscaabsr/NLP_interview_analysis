# NLP_interview_analysis

Core Analysis Pipeline

- **Preprocessing**: Moment identification and segmentation, text cleaning (including removal of questions, speaker labels, fillers, intonations, and punctuation marks), introduction removal, sentence splitting and grouping into answers (min 3 words sentences), duplicate removal;
 

## MOSAIC version

This analysis uses MOSAIC (topic modelling for first-person experiential reports):

- Repository: https://github.com/romybeaute/MOSAIC.git
- Branch: mosaic2.1
- Commit: e7a698abb4b4d68c3849393d399d24868d0d8589
- Commit date: 2026-04-23
- Recorded on: 2026-10-04

To get exactly this version:

```bash
git clone https://github.com/romybeaute/MOSAIC.git MOSAIC
git -C MOSAIC checkout e7a698abb4b4d68c3849393d399d24868d0d8589
```
Python 3.10.12
