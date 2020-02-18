from gensim import corpora, models
import jieba.posseg as jp, jieba

texts = [
    '美国教练坦言，没输给中国女排，是输给了郎平' * 99,
    '美国无缘四强，听听主教练的评价' * 99,
    '中国女排晋级世锦赛四强，全面解析主教练郎平的执教艺术' * 99,
    '为什么越来越多的人买MPV，而放弃SUV？跑一趟长途就知道了' * 99,
    '跑了长途才知道，SUV和轿车之间的差距' * 99,
    '家用的轿车买什么好' * 99]

if __name__ == "__main__":
    flags = ('n', 'nr', 'ns', 'nt', 'eng', 'v', 'd')  # 词性
    stopwords = ('没', '就', '知道', '是', '才', '听听', '坦言', '全面', '越来越', '评价', '放弃', '人')

    jieba.add_word("四强", 9, "n")
    words_ls = []

    for text in texts:
        words = [w.word for w in jp.cut(text) if w.flag in flags and w.word not in stopwords]
        words_ls.append(words)

    dictionary = corpora.Dictionary(words_ls)

    corpus = [dictionary.doc2bow(words) for words in words_ls]

    lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=2)

    print(lda.show_topic(0, 9999))
    print('概率总和', sum(i[1] for i in lda.show_topic(0, 9999)))
