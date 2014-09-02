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
            ('highschool', dict(overall=0.8, male=0.8, female=0.8, white=0.9,
                           black=0.7)),

            ]

        self.jobs = {
            'service': dict(highschool=0.5),
            'security': dict(highschool=0.7, prison=None),
            }
        self.job_commonality = {
            'service': 0.5,
            'security': 0.5,
            }

    def create_features(self):
        features = []
        features.append(self.pick_one(self.gender))
        features.append(self.pick_one(self.race))
        for feature, prob in self.probs:
            p = self.compute_conditional(features, prob)
            if self.rng.rand() < p:
                features.append(feature)
            else:
                features.append('no_' + feature)
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
        self.attributes = {}

class Employer:
    jobs_per_employer = 10
    def __init__(self, society):
        self.jobs = [Job(society) for i in range(self.jobs_per_employer)]

class Job:
    def __init__(self, society):
        self.employee = None
        self.society = society
        self.type = society.rng.choice(society.job_commonality.keys(),
                                       p=society.job_commonality.values())

    def compute_suitability(self, person):
        total = 0
        for feature, value in self.society.jobs[self.type].items():
            if value is None and feature in person.features:
                return None
            if feature in person.features:
                total += value
            elif feature in person.attributes:
                total += value * person.attributes[feature]
        return total




if __name__ == '__main__':
    rng = np.random.RandomState()
    society = Society(rng)
    people = [Person(society) for i in range(10)]
    job = Job(society)
    print job.type
    for p in people:
        print p.features, job.compute_suitability(p)




