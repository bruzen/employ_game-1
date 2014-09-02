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
            #('prison', dict(overall=0.4, male=0.2, female=0.1, white=0.2,
            #               black=0.3, black_male=0.9)),
            ('prison', dict(overall=0.3)),
            ('highschool', dict(overall=0.8, male=0.8, female=0.8, white=0.9,
                           black=0.7)),

            ]

        self.jobs = {
            'service': dict(highschool=0.5),
            'security': dict(highschool=0.7, prison=None, experience=0.5),
            }
        self.job_commonality = {
            'service': 0.7,
            'security': 0.3,
            }
        self.job_income = {
            'service': (20000, 1000),   # starting, annual raise
            'security': (40000, 2000),
        }
        self.job_retention = {
            'service': [0.2, 0.6],
            'security': [0.7, 0.8],
            }
        self.job_productivity = {
            'service': (50000, 1),   # starting, annual raise
            'security': (80000, 0.5),
        }
        self.job_hiring = {
            'service': 5000,   # starting, annual raise
            'security': 20000,
        }

        self.neighbourhoods = [Neighbourhood(self)
                               for i in range(self.neighbourhood_count)]

    # numerical
    def create_attributes(self):
        attr = {}
        attr['prob_apply'] = self.rng.normal(0.5, 0.25)
        attr['distance_penalty'] = self.rng.uniform(0, 0.5)
        attr['interview_skill'] = self.rng.normal(0.2, 0.2)
        attr['interview_skill_sd'] = self.rng.uniform(0.1, 0.4)
        attr['experience'] = 0.0
        return attr

    # binary
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
        if len(relevant) == 0:
            return pF
        p = [prob[k] for k in relevant]
        if len(p) == 1:
            return p
        all = np.prod(p)
        none = np.prod([1-pp for pp in p])
        probability = all / (all + (pF / (1-pF)) * none)
        return probability


class Person:
    def __init__(self, society):
        self.society = society
        self.age = 16
        self.job = None
        self.income = 0
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
        self.society = society
        self.jobs = [Job(society, self) for i in range(self.jobs_per_employer)]
        self.neighbourhood = society.rng.choice(society.neighbourhoods)
        self.total_hiring_cost = 0
        self.total_salary = 0
        self.total_productivity = 0
        self.total_net = 0

    def step(self, dt):
        self.hiring_cost = 0
        self.salary = 0
        self.productivity = 0

        for j in self.jobs:
            if j.employee is not None:
                start, slope = self.society.job_income[j.type]
                salary = start + j.employee.job_length * slope
                self.salary += salary * dt
                j.employee.income += salary * dt

                prod_max, prod_time = self.society.job_productivity[j.type]
                prod = prod_max * (1-np.exp(-j.employee.job_length/prod_time))
                self.productivity += prod * dt

                if j.employee.job_length == 0.0:
                    self.hiring_cost += self.society.job_hiring[j.type]

            self.net = self.productivity - self.hiring_cost - self.salary
        self.total_net += self.net
        self.total_salary += self.salary
        self.total_productivity += self.productivity
        self.total_hiring_cost += self.hiring_cost

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
    employer_count = 10
    people_per_step = 1
    years_per_step = 0.1
    max_age = 25

    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed=seed)
        self.society = Society(self.rng)
        self.employers = [Employer(self.society)
                          for i in range(self.employer_count)]
        self.people = []
        self.steps = 0
        self.interventions = []

    def step(self):
        self.steps += 1
        for interv in self.interventions:
            interv.apply(self, self.steps)

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

        interview = {}
        for job, applicants in applications.items():
            for a in applicants:
                score = job.compute_suitability(a)
                score += self.rng.normal(a.attributes['interview_skill'],
                                         a.attributes['interview_skill_sd'])
                interview[(job, a)] = score

        iterations = 10
        for i in range(iterations):
            all_offers = {}
            for job, applicants in applications.items():

                score = [interview[(job, a)] for a in applicants]

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
                        person.job_evaluation = 'medium'
                        person.job_length = 0.0

        for e in self.employers:
            e.step(self.years_per_step)
        self.increase_age()
        self.remove_older()
        self.job_evaluation()

    def job_evaluation(self):
        for p in self.people:
            if p.job is not None:
                index = int(p.job_length)
                retention = self.society.job_retention[p.job.type]
                if index >= len(retention):
                    r = retention[-1]
                else:
                    r = retention[index]
                r = (1-r) * self.years_per_step
                if self.society.rng.rand() < r:
                    self.fire(p)

    def fire(self, person):
        assert person.job is not None
        person.job.employee = None
        person.job = None



    def increase_age(self):
        for p in self.people:
            p.age += self.years_per_step
            if p.job is not None:
                p.job_length += self.years_per_step
                p.attributes['experience'] += self.years_per_step

    def remove_older(self):
        for p in self.people:
            if p.age > self.max_age:
                self.people.remove(p)
                if p.job is not None:
                    p.job.employee = None


    def calc_employment(self):
        count = 0
        for p in self.people:
            if p.job is not None:
                count += 1
        return float(count)/len(self.people)


    def check_jobs(self):
        for p in self.people:
            print p.features, p.job.type if p.job is not None else None
        #for e in self.employers:
        #    print e.total_net






class HighschoolCertificateIntervention:
    def __init__(self, time, proportion):
        self.time = time
        self.proportion = proportion

    def apply(self, model, timestep):
        if timestep == self.time:
            for p in model.people:
                if 'no_highschool' in p.features:
                    if model.rng.rand() < self.proportion:
                        p.features.append('highschool')
                        p.features.remove('no_highschool')
        else:
            for p in model.people:
                if p.age == 16:
                    if 'no_highschool' in p.features:
                        if model.rng.rand() < self.proportion:
                            p.features.append('highschool')
                            p.features.remove('no_highschool')





if __name__ == '__main__':

    m = Model()

    m.interventions.append(HighschoolCertificateIntervention(50, 1.0))
    m.step()

    for i in range(1000):
        print i, len(m.people), m.calc_employment()
        m.check_jobs()
        m.step()




