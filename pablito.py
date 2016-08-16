from collections import defaultdict

class NGram(object):

    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        assert n > 0
        self.n = n
        self.counts = counts = defaultdict(int)

        sents = list(map((lambda x: ['<s>']*(n-1) + x), sents))
        sents = list(map((lambda x: x + ['</s>']), sents))

        for sent in sents:
            for i in range(len(sent) - n + 1):
                ngram = tuple(sent[i: i + n])
                counts[ngram] += 1
                counts[ngram[:-1]] += 1


class NGramGenerator(object):

    def __init__(self, model):
        """
        model -- n-gram model.
        """
        self.n = model.n
        self.probs = probs = dict()
        self.sorted_probs = dict()
        # pre, list of grams with length n-1 (of a n-gram model)
        pre = [elem for elem in model.counts.keys() if not len(elem) == self.n]
        # suf, list of grams with length n (of a n-gram model)
        suf = [elem for elem in model.counts.keys() if len(elem) == self.n]

        for elem in suf:
            prfx = elem[:-1]
            sfx = elem[-1]
            # if prfx already in dict, we add the new sufix and its
            # probability and update the dict
            if prfx in probs:
                aux = probs[prfx]
                # probs values are dicts with (token, cond_prob of token)
                probs[prfx] = {sfx: model.cond_prob(sfx, prfx)}
                probs[prfx].update(aux)
            else:
                probs[prfx] = {sfx: model.cond_prob(sfx, prfx)}
        # order the dict by its values with higher probability
        # so we can use the inverse transform method
        sp = [list(probs[x].items()) for x in pre]
        self.sorted_probs = {
            pre[i]: sorted(sp[i], key=lambda x:
                           (-x[1], x[0])) for i in range(len(sp))
        }

    def generate_sent(self):
        """Randomly generate a sentence."""
        n = self.n
        sent = ('<s>',)*(n-1)
        if n == 1:
            sent = ()
        # generate until STOP symbol comes up
        while '</s>' not in sent:
            sent += (self.generate_token(sent[-n+1:]),)
        return sent[n-1:-1]

    def generate_token(self, prev_tokens=None):
        """Randomly generate a token, given prev_tokens.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        n = self.n
        if n == 1:
            prev_tokens = tuple()
        p = random()
        res = ''
        choices = self.sorted_probs[prev_tokens]
        # applying the inverse transform method
        acc = choices[0][1]
        for i in range(0, len(choices)):
            if p < acc:
                res = choices[i][0]
                break
            else:
                acc += choices[i][1]
        return res
