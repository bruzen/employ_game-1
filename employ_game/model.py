import numpy as np
from collections import OrderedDict

class Society:
    neighbourhood_rows = 2
    neighbourhood_cols = 2
    neighbourhood_count = neighbourhood_rows * neighbourhood_cols

    childcare_cost = 4000
    transportation_cost = 2000

    def __init__(self, rng):
        self.rng = rng
        self.gender = OrderedDict(male=0.5, female=0.5)
        self.race = OrderedDict(black=0.3, white=0.3, hispanic=0.2, asian=0.2)

        self.interv_private = 0
        self.interv_public = 0

        # NOTE: order matters!  probabilities can be conditional on anything
        #       that is above them in the list below
        # keys:   overall is p(F)
        #         other terms are p(F|term)
        #         underscores_like_this are p(F|underscores^like^this)
        self.probs = [
            #('prison', dict(overall=0.4, male=0.2, female=0.1, white=0.2,
            #               black=0.3, black_male=0.9)),
            ('prison', OrderedDict(overall=0.02)),
            ('childcare', OrderedDict(overall=0.4)),
            ('highschool', OrderedDict(overall=0.4)),
            ]

        self.jobs = {
            'service_low': dict(highschool=0.5, experience_service=1),
            'service_high': dict(no_highschool=None, experience_service=1, baseline=-3),
            'manufacturing_low': dict(highschool=0.5, experience_manufacturing=1),
            'manufacturing_high': dict(no_highschool=None, experience_manufacturing=1, baseline=-3),
            }
        self.job_sector = {
            'service_low': 'service',
            'service_high': 'service',
            'manufacturing_high': 'manufacturing',
            'manufacturing_low': 'manufacturing',
            }
        self.job_commonality = OrderedDict(service_low=0.3, service_high=0.3,
                                           manufacturing_low=0.2, manufacturing_high=0.2)
        self.job_income = {
            'service_low': (13000, 1300),   # starting, annual raise
            'service_high': (22000, 3600),   # starting, annual raise
            'manufacturing_low': (14000, 900),   # starting, annual raise
            'manufacturing_high': (27000, 3500),   # starting, annual raise
        }
        self.job_retention = {
            'service_low': [0.5, 0.8],    # first year, next years
            'service_high': [0.5, 0.8],
            'manufacturing_low': [0.5, 0.8],
            'manufacturing_high': [0.5, 0.8],
            }
        self.job_productivity = {
            # exponential curves take approx 3*exp_time to reach asymptote
            'service_low': (22000, 0.5),    # final asymptote, exp_time
            'service_high': (57000, 1),    # final asymptote, exp_time
            'manufacturing_low': (27000, 0.5),    # final asymptote, exp_time
            'manufacturing_high': (63000, 1),    # final asymptote, exp_time
        }
        self.job_hiring = {
            'service_low': 5000,
            'service_high': 5000,
            'manufacturing_low': 5000,
            'manufacturing_high': 5000,
        }

        self.neighbourhoods = [Neighbourhood(self)
                               for i in range(self.neighbourhood_count)]
        self.distance_penalty_scale = 10000
        self.set_racial_discrimination(0.3)

    def set_racial_discrimination(self, value=1.0,
                                  races=['black', 'hispanic', 'asian']):
        for v in self.jobs.values():
            for r in races:
                v[r] = -value
    def adjust_retention(self, value):
        for v in self.job_retention.values():
            for i in range(len(v)):
                v[i] = 0.5 * (v[i] + value)




    # numerical
    def create_attributes(self):

        attr = {}
        attr['prob_apply'] = self.rng.normal(0.5, 0.25)
        attr['distance_penalty'] = self.rng.uniform(0, 0.5)
        attr['distance_penalty'] = self.rng.uniform(1.0, 2.0)
        attr['interview_skill'] = self.rng.normal(0.2, 0.2)
        attr['interview_skill_sd'] = self.rng.uniform(0.1, 0.4)
        attr['experience'] = 0.0
        attr['experience_service'] = 0.0
        attr['experience_manufacturing'] = 0.0
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
        assert isinstance(options, OrderedDict)
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

def color_blend(c1, c2, blend):
    if blend <= 0:
        c = c1
    elif blend >= 1.0:
        c = c2
    else:
        c = c1 * (1-blend) + c2 * (blend)
    return '#%02x%02x%02x' % (c[0]*255, c[1]*255, c[2]*255)

class Person:
    def __init__(self, society):
        self.society = society
        self.age = 16
        self.job = None
        self.job_length = 0.0
        self.income = 0
        self.features = society.create_features()
        self.attributes = society.create_attributes()
        self.neighbourhood = society.rng.choice(society.neighbourhoods)
        self.location = self.neighbourhood.allocate_location()
        self.local_preference_bonus = 1000
        self.childcare_support = 0

    def does_apply(self, job):
        salary = self.society.job_income[job.type][0]
        reservation_wage = 11000
        if 'childcare' in self.features:
            reservation_wage += self.society.childcare_cost
            reservation_wage -= self.childcare_support
        if self.neighbourhood is not job.employer.neighbourhood:
            reservation_wage += self.society.transportation_cost
        if reservation_wage > salary:
            return False


        p = self.attributes['prob_apply']
        if job.employer.neighbourhood is not self.neighbourhood:
            p -= self.attributes['distance_penalty'] * self.society.distance_penalty_scale
        return self.society.rng.rand() < p

    def compute_suitability(self, job):
        score = 0
        if self.neighbourhood is job.employer.neighbourhood:
            score += self.local_preference_bonus
        else:
            score -= self.society.transportation_cost
        score += self.society.job_income[job.type][0]
        return score

    def get_color(self):
        if self.job is None:
            return color_blend(np.array([1.0, 0.5, 0.5]), np.array([1.0, 0.0, 0.0]),
                               self.job_length/5.0)
        else:
            return color_blend(np.array([0.5, 0.5, 1.0]), np.array([0.0, 0.0, 1.0]),
                               self.job_length/5.0)

    def get_info(self):
        text = '<h1>Person:</h1>'
        visible = [x[0] for x in self.society.probs]
        text += ', '.join([f for f in self.features if f in visible])
        text += '<br/>%3.1f years old' % self.age
        status = 'employed' if self.job is not None else 'unemployed'
        text +='<br/>%s for %2.1f years' % (status, self.job_length)
        if self.job: text += ' at %s' % self.job.type
        text +='<br/>experience: %2.1f years (%2.1f service, %2.1f manufacturing)' % (self.attributes['experience'],
                                                                                      self.attributes['experience_service'],
                                                                                      self.attributes['experience_manufacturing'])
        return text



class Employer:
    jobs_per_employer = 10
    def __init__(self, society):
        self.society = society
        self.jobs = [Job(society, self) for i in range(self.jobs_per_employer)]
        self.neighbourhood = society.rng.choice(society.neighbourhoods)
        self.location = self.neighbourhood.allocate_location()
        self.total_hiring_cost = 0
        self.total_salary = 0
        self.total_productivity = 0
        self.total_net = 0

    def get_color(self):
        return '#888'
    def get_info(self):
        text = '<h1>Employer:</h1>'
        text += 'net: <strong>$%5.2f</strong>' % self.total_net
        return text

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
            if feature == 'baseline':
                total += value
            elif feature in person.features:
                total += value
            elif feature in person.attributes:
                total += value * person.attributes[feature]
        return total




class Neighbourhood:
    def __init__(self, society):
        self.society = society
        self.rows = 7
        self.cols = 7
        x, y = np.meshgrid(np.arange(self.cols), np.arange(self.rows))
        self.locations = zip(x.flatten(),y.flatten())
    def allocate_location(self):
        if len(self.locations) == 0:
            print 'warning: not enough space in neighbourhood'
            return (0,0)
        index = self.society.rng.randint(len(self.locations))
        return self.locations.pop(index)
    def free_location(self, loc):
        self.locations.append(loc)


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
        self.data = {}
        self.init_data()

    def step(self):
        self.steps += 1
        for interv in self.interventions:
            interv.apply(self, self.steps)

        for i in range(self.people_per_step):
            self.people.append(Person(self.society))


        applications = OrderedDict()
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
            all_offers = OrderedDict()
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

        self.update_data()

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
        person.job_length = 0.0



    def increase_age(self):
        for p in self.people:
            p.age += self.years_per_step
            p.job_length += self.years_per_step
            if p.job is not None:
                p.attributes['experience'] += self.years_per_step
                sector = self.society.job_sector[p.job.type]
                p.attributes['experience_%s' % sector] += self.years_per_step
            p.attributes['age'] = p.age

    def remove_older(self):
        for p in self.people:
            if p.age > self.max_age:
                p.neighbourhood.free_location(p.location)
                self.people.remove(p)
                if p.job is not None:
                    p.job.employee = None


    def calc_employment(self):
        count = 0
        for p in self.people:
            if p.job is not None:
                count += 1
        return float(count)/len(self.people)

    def calc_feature_employment(self, feature):
        count = 0
        total = 0
        for p in self.people:
            if feature in p.features:
                total += 1
                if p.job is not None:
                    count += 1
        return float(count) / total

    def calc_attribute_employment(self, attribute, threshold):
        count = 0
        total = 0
        for p in self.people:
            if p.attributes[attribute] >= threshold:
                total += 1
                if p.job is not None:
                    count += 1
        return float(count) / total

    def calc_feature_rate(self, feature):
        count = 0
        for p in self.people:
            if feature in p.features:
                count += 1
        return float(count)/len(self.people)

    def calc_attribute_rate(self, attribute, threshold):
        count = 0
        for p in self.people:
            if p.attributes[attribute] >= threshold:
                count += 1
        return float(count)/len(self.people)

    def calc_employer_net(self):
        return sum([e.net for e in self.employers])


    def check_jobs(self):
        for p in self.people:
            print p.features, p.job.type if p.job is not None else None
        #for e in self.employers:
        #    print e.total_net

    def init_data(self):
        self.data['employment'] = []
        self.data['employer_net'] = []
        self.data['highschool'] = []
        self.data['employment_childcare'] = []
        self.data['employment_nohighschool'] = []
        self.data['employment_2_or_more_years'] = []
        self.data['employment_18plus'] = []
        self.data['proportion_childcare'] = []
        self.data['proportion_nohighschool'] = []
        self.data['proportion_2_or_more_years'] = []
        self.data['proportion_18plus'] = []

        self.data['cost_hiring'] = []
        self.data['cost_salary'] = []
        self.data['production'] = []
        self.data['interv_public'] = []
        self.data['interv_private'] = []

        #for race in self.society.race.keys():
        #    self.data['employment_%s' % race] = []
        #    self.data['proportion_%s' % race] = []

    def update_data(self):
        if self.steps >= 100:
            self.data['employment'].append(self.calc_employment()*100)
            self.data['employer_net'].append(self.calc_employer_net()*0.001)
            self.data['highschool'].append(self.calc_feature_rate('highschool')*100)
            self.data['employment_childcare'].append(self.calc_feature_employment('childcare')*100)
            self.data['employment_nohighschool'].append(self.calc_feature_employment('no_highschool')*100)
            self.data['employment_2_or_more_years'].append(self.calc_attribute_employment('experience', threshold=2.0)*100)
            self.data['employment_18plus'].append(self.calc_attribute_employment('age', threshold=18)*100)
            self.data['proportion_childcare'].append(self.calc_feature_rate('childcare')*100)
            self.data['proportion_nohighschool'].append(self.calc_feature_rate('no_highschool')*100)
            self.data['proportion_2_or_more_years'].append(self.calc_attribute_rate('experience', threshold=2.0)*100)
            self.data['proportion_18plus'].append(self.calc_attribute_rate('age', threshold=18)*100)
            self.data['cost_hiring'].append(sum([e.hiring_cost for e in self.employers]))
            self.data['cost_salary'].append(sum([e.salary for e in self.employers]))
            self.data['production'].append(sum([e.productivity for e in self.employers]))
            self.data['interv_public'].append(self.society.interv_public)
            self.data['interv_private'].append(self.society.interv_private)
            #for race in self.society.race.keys():
            #    self.data['employment_%s' % race].append(self.calc_feature_employment(race)*100)
            #    self.data['proportion_%s' % race].append(self.calc_feature_rate(race)*100)

    def get_grid(self):
        grid = []
        for e in self.employers:
            x, y = self.get_location(e)
            color = e.get_color()
            item = dict(type='employer', x=x, y=y, color=color, info=e.get_info())
            grid.append(item)
        for p in self.people:
            x, y = self.get_location(p)
            color = p.get_color()
            item = dict(type='person', x=x, y=y, color=color, info=p.get_info())
            grid.append(item)
        return grid

    def get_location(self, item):
        xx, yy = item.location
        n_index = self.society.neighbourhoods.index(item.neighbourhood)
        x = (n_index % self.society.neighbourhood_cols) * item.neighbourhood.cols + xx
        y = (n_index / self.society.neighbourhood_cols) * item.neighbourhood.rows + yy
        return x, y

    def get_data(self):
        self.data['grid'] = self.get_grid()
        return self.data

class RetentionIntervention:
    def __init__(self, time, value):
        self.time = time
        self.value = value
    def apply(self, model, timestep):
        if timestep == self.time:
            model.society.adjust_retention(self.value)




class SocietyParameterIntervention:
    def __init__(self, time, parameter, value):
        self.time = time
        self.parameter = parameter
        self.value = value
    def apply(self, model, timestep):
        if timestep == self.time:
            #print ('setting', self.parameter, self.value,
            #       'from', getattr(model.society, self.parameter))
            setattr(model.society, self.parameter, self.value)


class DiscriminationIntervention:
    def __init__(self, time, value):
        self.time = time
        self.value = value
    def apply(self, model, timestep):
        if timestep == self.time:
            #print ('setting', self.parameter, self.value,
            #       'from', getattr(model.society, self.parameter))
            model.society.set_racial_discrimination(self.value)

class ChildcareIntervention:
    def __init__(self, time, proportion, value, cost_sunk, cost_fixed, cost_variable, public_proportion):
        self.time = time
        self.proportion = proportion
        self.value = value
        self.cost_sunk = cost_sunk
        self.cost_fixed = cost_fixed
        self.cost_variable = cost_variable
        self.public_proportion = public_proportion
    def apply(self, model, timestep):
        if timestep == self.time:
            model.society.interv_public += self.public_proportion * self.cost_sunk
            model.society.interv_private += (1-self.public_proportion) * self.cost_sunk

            for p in model.people:
                if 'childcare' in p.features:
                    if model.rng.rand() < self.proportion:
                        p.childcare_support = self.value
                    else:
                        p.childcare_support = 0
        elif timestep > self.time:
            model.society.interv_public += self.public_proportion * self.cost_fixed * Model.years_per_step
            model.society.interv_private += (1-self.public_proportion) * self.cost_fixed * Model.years_per_step

            for p in model.people:
                if p.age < 16 + Model.years_per_step * 2:
                    if 'childcare' in p.features:
                        if model.rng.rand() < self.proportion:
                            model.society.interv_public += self.public_proportion * self.cost_variable * Model.years_per_step
                            model.society.interv_private += (1-self.public_proportion) * self.cost_variable * Model.years_per_step
                            p.childcare_support = self.value
                        else:
                            p.childcare_support = 0



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
        elif timestep > self.time:
            for p in model.people:
                if p.age < 16 + Model.years_per_step * 2:
                    if 'no_highschool' in p.features:
                        if model.rng.rand() < self.proportion:
                            p.features.append('highschool')
                            p.features.remove('no_highschool')

def memoize(f):
    """ Memoization decorator for functions taking one or more arguments. """
    class memodict(dict):
        def __init__(self, f):
            self.f = f
        def __call__(self, *args):
            return self[args]
        def __missing__(self, key):
            ret = self[key] = self.f(*key)
            return ret
    return memodict(f)


model_cache = {}
import copy

def find_cached_model(seed, actions):
    for step in reversed(range(len(actions)+1)):
        result = model_cache.get((seed, tuple(actions[:step])), None)
        if result is not None:
            step, model = result
            return step, copy.deepcopy(model)
    model = Model(seed=seed)
    presteps = 100
    for i in range(presteps):
        model.step()
    model_cache[(seed, ())] = copy.deepcopy(model)
    return -1, model

@memoize
def run(seed, *actions):
    step, model = find_cached_model(seed, actions)
    presteps = 100
    steps_per_action = 10
    for i, action in enumerate(actions):
        if i > step:  # if we haven't done this step yet
            interv_step = presteps + 1 + steps_per_action * i
            if action == 'hs_diploma':
                interv = HighschoolCertificateIntervention(interv_step, 1.0)
                model.interventions.append(interv)
            elif action == 'mobility+':
                interv = SocietyParameterIntervention(interv_step,
                                        'distance_penalty_scale', 0)
            elif action == 'mobility-':
                interv = SocietyParameterIntervention(interv_step,
                                        'distance_penalty_scale', 10000.0)
            elif action == 'discriminate-normal':
                interv = DiscriminationIntervention(interv_step, 0.3)
            elif action == 'discriminate-high':
                interv = DiscriminationIntervention(interv_step, 2.0)
            elif action == 'discriminate-low':
                interv = DiscriminationIntervention(interv_step, 0.0)
            elif action == 'retention+':
                interv = RetentionIntervention(interv_step, 1.0)
            elif action == 'retention-':
                interv = RetentionIntervention(interv_step, 0.0)
            elif action == 'childcare-low':
                interv = ChildcareIntervention(interv_step, 0.2, 4000, cost_sunk=20000, cost_fixed=5000, cost_variable=1000, public_proportion=0.5)
            elif action == 'childcare-med':
                interv = ChildcareIntervention(interv_step, 0.5, 4000, cost_sunk=40000, cost_fixed=15000, cost_variable=1000, public_proportion=0.5)
            elif action == 'childcare-high':
                interv = ChildcareIntervention(interv_step, 0.8, 4000, cost_sunk=80000, cost_fixed=15000, cost_variable=1000, public_proportion=0.5)
            else:
                interv = None
                print 'unknown intervention', action
            if interv is not None:
                model.interventions.append(interv)
            for ii in range(steps_per_action):
                model.step()
            model_cache[(seed, tuple(actions[:(i+1)]))] = (i, copy.deepcopy(model))
    return model.get_data()






if __name__ == '__main__':
    print run(1, 'init')['employment_black']
    1/0



    m = Model()

    m.interventions.append(HighschoolCertificateIntervention(50, 1.0))
    m.step()

    for i in range(1000):
        print i, len(m.people), m.calc_employment()
        m.check_jobs()
        m.step()




