PROJECT_PATH = './main'
RESULT_PATH = F'{PROJECT_PATH}/results'

SURPRISE_LANGUAGES = ['de', 'yo']
NEW_LANGUAGES = ['es', 'fa', 'fr', 'hi', 'zh'] + SURPRISE_LANGUAGES
LANGUAGES = ['ar', 'bn', 'en', 'es', 'fa', 'fi', 'fr', 'hi', 'id', 'ja', 'ko', 'ru', 'sw', 'te', 'th', 'zh'] + SURPRISE_LANGUAGES

CORPUS_NAME = 'miracl/miracl-corpus'
DATASET_PATHS = {
    lang: {
        'dev': [
            f'{PROJECT_PATH}/miracl/miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-dev.tsv',
            f'{PROJECT_PATH}/miracl/miracl-v1.0-{lang}/qrels/qrels.miracl-v1.0-{lang}-dev.tsv',
        ],
        'testB': [
            f'{PROJECT_PATH}/miracl/miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-test-b.tsv',
        ],
    } for lang in LANGUAGES
}
for lang in LANGUAGES:
    if lang not in SURPRISE_LANGUAGES:
        DATASET_PATHS[lang]['train'] = [
            f'{PROJECT_PATH}/miracl/miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-train.tsv',
            f'{PROJECT_PATH}/miracl/miracl-v1.0-{lang}/qrels/qrels.miracl-v1.0-{lang}-train.tsv',
        ]
    if lang not in NEW_LANGUAGES:
        DATASET_PATHS[lang]['testA'] = [
            f'{PROJECT_PATH}/miracl/miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-test-a.tsv',
        ]