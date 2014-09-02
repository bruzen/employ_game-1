import numpy as np

class Society:
    neighbourhood_count = 3
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
        self.job_income = {
            'service': (20000, 1000),   # starting, annual raise
            'security': (40000, 2000),
        }

        self.neighbourhoods = [Neighbourhood(self)
                               for i in range(self.neighbourhood_count)]

    def create_attributes(self):
        attr = {}
        attr['prob_apply'] = self.rng.normal(0.5, 0.25)
        attr['distance_penalty'] = self.rng.uniform(0, 0.5)
        attr['interview_skill'] = self.rng.normal(0.2, 0.2)
        attr['interview_skill_sd'] = self.rng.uniform(0.1, 0.4)
        return attr

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
        self.society = society
        self.age = 16
        self.job = None
        self.features = society.create_features()
        self.attributes = society.create_attributes()
        self.neighbourhood = society.rng.choice(society.neighbourhoods)

    def does_apply(self, job):
        p = self.attributes['prob_apply']
        if job.employer.neighbourhood is not self.neighbourhood:
            p -= self.attributes['distance_penalty']
        return self.society.rng.rand() < p

    def compute_suitability(self, job):
        score = 0
        if self.neighbourhood is job.employer.neighbourhood:
            score += 5000
        score += self.society.job_income[job.type][0]
        return score




class Employer:
    jobs_per_employer = 10
    def __init__(self, society):
        self.jobs = [Job(society, self) for i in range(self.jobs_per_employer)]
        self.neighbourhood = society.rng.choice(society.neighbourhoods)

class Job:
    def __init__(self, society, employer):
        self.employee = None
        self.employer = employer
        self.society = society
        self.type = society.rng.choice(society.job_commonality.keys(),
                                       p=society.job_commonality.values())

    def compute_suitability(self, person):
        total = 0
        for feature, value in self.society.jobs[self.type].items():
            if value is None and feature in person.features:
                return -np.inf
            if feature in person.features:
                total += value
            elif feature in person.attributes:
                total += value * person.attributes[feature]
        return total

class Neighbourhood:
    def __init__(self, society):
        self.society = society


class Model:
    employer_count = 100
    people_per_step = 10
    years_per_step = 0.1
    max_age = 25

    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed=seed)
        self.society = Society(self.rng)
        self.employers = [Employer(self.society)
                          for i in range(self.employer_count)]
        self.people = []

    def step(self):
        for i in range(self.people_per_step):
            self.people.append(Person(self.society))

        applications = {}
        for e in self.employers:
            for j in e.jobs:
                if j.employee is None:
                    applications[j] = []
        for p in self.people:
            if p.job is None:
                for j in applications.keys():
                    if p.does_apply(j):
                        applications[j].append(p)

        iterations = 10
        for i in range(iterations):
            all_offers = {}
            for job, applicants in applications.items():

                #TODO: optimize this so it isn't computed each iteration
                score = [job.compute_suitability(a) for a in applicants]

                for i,a in enumerate(applicants):
                    score[i] += self.rng.normal(a.attributes['interview_skill'],
                                                a.attributes['interview_skill_sd'])

                if len(score) > 0:
                    max_score = max(score)
                    if max_score > 0:
                        index = score.index(max_score)
                        person = applicants[index]
                        if person not in all_offers:
                            all_offers[person] = []
                        all_offers[person].append(job)

            for person, offers in all_offers.items():
                score = [person.compute_suitability(j) for j in offers]
                if len(score) > 0:
                    max_score = max(score)
                    if max_score > 0:
                        index = score.index(max_score)
                        job = offers[index]

                        person.job = job
                        job.employee = person

    def calc_employment(self):
        count = 0
        for p in self.people:
            if p.job is not None:
                count += 1
        return float(count)/len(self.people)









if __name__ == '__main__':

    m = Model()
    m.step()
    for i in range(100):
        print i, len(m.people), m.calc_employment()
        m.step()




