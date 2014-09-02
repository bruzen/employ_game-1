import numpy as np

class Society:
    def __init__(self, rng):
        self.rng = rng
        self.gender = dict(male=0.5, female=0.5)
        self.race = dict(black=0.3, white=0.3, hispanic=0.2, asian=0.2)

        # NOTE: order matters!  probabilities can be conditional on anything
        #       that is above them in the list below
        # keys:   overall is p(F)
        #         other terms are p(F|term)
        #         underscores_like_this are p(F|underscores^like^this)
        self.probs = [
            ('prison', dict(overall=0.4, male=0.2, female=0.1, white=0.2,
                           black=0.3, black_male=0.9)),

            ]

    def create_features(self):
        features = []
        features.append(self.pick_one(self.gender))
        features.append(self.pick_one(self.race))
        for feature, prob in self.probs:
            p = self.compute_conditional(features, prob)
            if self.rng.rand() < p:
                features.append(feature)
        return features

    def pick_one(self, options):
        return self.rng.choice(options.keys(), p=options.values())

    def compute_conditional(self, feature, prob):
        # given the set of conditional probabilities and the given features,
        # return the probability of having the feature

        #TODO: validate this algorithm
        relevant = [k for k in prob.keys() if '_' not in k and k in feature]
        for k in prob.keys():
            if '_' in k:
                parts = k.split('_')
                for p in parts:
                    if p not in feature:
                        break
                else:
                    for p in parts:
                        if p in relevant:
                            relevant.remove(p)
                    relevant.append(k)
        pF = prob['overall']
        p = [prob[k] for k in relevant]
        all = np.prod(p)
        none = np.prod([1-pp for pp in p])
        probability = all / (all + (pF / (1-pF)) * none)
        return probability






class Person:
    def __init__(self, society):
        self.rng = rng
        self.age = 16
        self.job = None
        self.features = society.create_features()





if __name__ == '__main__':
    rng = np.random.RandomState()
    society = Society(rng)
    people = [Person(society) for i in range(10)]
    for p in people:
        print p.features


