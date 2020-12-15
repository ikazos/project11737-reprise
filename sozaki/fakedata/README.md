#   Fake WALS / SIGTYP data

The data is fake. Use this only for testing purposes.

##  `wals/parameters.csv`

3 parameters:

*   1A: Parameter 1A
*   2A: Parameter 2A
*   2B: Parameter 2B

##  `wals/languages.csv`

3 languages:

*   Japanese
*   Korean
*   Mandarin

For `languages.csv`, the data are taken from real WALS data (`realdata/wals/languages.csv`).

##  `wals/codes.csv`

Each parameter has the following values:

*   1A: Parameter 1A
    -   1A-1: Value 1A-1
    -   1A-2: Value 1A-2
*   2A: Parameter 2A
    -   2A-1: Value 2A-1
    -   2A-2: Value 2A-2
    -   2A-3: Value 2A-3
*   2B: Parameter 2B
    -   2B-1: Value 2B-1
    -   2B-2: Value 2B-2
    -   2B-3: Value 2B-3
    -   2B-4: Value 2B-4

##  `sigtyp/train.csv`

Japanese:

*   1A: 1
*   2A: 2

##  `sigtyp/dev.csv`

Korean:

*   1A: 2
*   2B: 3

##  `sigtyp/test_blinded.csv`

Mandarin:

*   1A: 1
*   2A: ?
*   2B: ?

##  `test/{ src-train.txt, tgt-train.txt, src-val.txt, tgt-val.txt }`

In this experiment we translate between Japanese, Korean and Mandarin.

The language ID tokens are LANG_JPN, LANG_KOR and LANG_CMN. Notice the Mandarin token is not LANG_MND (the WALS language ID), but LANG_CMN (ISO639P3).

In training and dev:

*   Sentence 1: Japanese -> Korean
*   Sentence 2: Japanese -> Mandarin
*   Sentence 3: Korean -> Japanese
*   Sentence 4: Korean -> Mandarin
*   Sentence 5: Mandarin -> Japanese
*   Sentence 6: Mandarin -> Korean