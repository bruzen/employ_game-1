import os.path
import json
import random
import uuid as uuid_package

import pkgutil
import employ_game

import model


from collections import OrderedDict

actions = OrderedDict()
action_texts = OrderedDict()
seeds = {}
names = {}

class Server(employ_game.swi.SimpleWebInterface):

    def clear(self):
        actions.clear()
        action_texts.clear()
        seeds.clear()
        names.clear()
        model.clear_cache()

    def swi_static(self, *path):
        if self.user is None: return
        fn = os.path.join('static', *path)
        if fn.endswith('.js'):
            mimetype = 'text/javascript'
        elif fn.endswith('.css'):
            mimetype = 'text/css'
        elif fn.endswith('.png'):
            mimetype = 'image/png'
        elif fn.endswith('.jpg'):
            mimetype = 'image/jpg'
        elif fn.endswith('.gif'):
            mimetype = 'image/gif'
        elif fn.endswith('.otf'):
            mimetype = 'font/font'
        else:
            raise Exception('unknown extenstion for %s' % fn)

        data = pkgutil.get_data('employ_game', fn)
        return (mimetype, data)

    def swi_favicon_ico(self):
        icon = pkgutil.get_data('employ_game', 'static/favicon.ico')
        return ('image/ico', icon)

    #def swi(self):
    #    if self.user is None:
    #        return self.create_login_form()
    #    html = pkgutil.get_data('employ_game', 'templates/index.html')
    #    return html# % dict(uuid=uuid_package.uuid4())

    def swi(self):
        if self.user is None:
            return self.create_login_form()
        html = pkgutil.get_data('employ_game', 'templates/overview.html')
        return html

    def swi_overview_json(self, command=None):

        if command == 'clear':
            self.clear()

        maximum = 10
        substeps = 10

        uuids = [k for k in actions.keys() if len(actions[k])>1]

        data = {k: self.run_game(k) for k in uuids}

        #time = []
        #for i, uuid in enumerate(uuids):
        #    color = ['blue', 'red', 'green', 'magenta', 'cyan', 'black'][i % 6]
        #    key = names[uuid]
        #    values = []
        #    for j in range(len(data[uuid]['employment'])):
        #        values.append(dict(x=float(j)/substeps, y=data[uuid]['employment'][j]))
        #    values.append(dict(x=maximum, y=None))
        #    time.append(dict(values=values, key=key, color=color, area=False))

        bar = []
        for i, uuid in enumerate(uuids):
            color = ['blue', 'red', 'green', 'magenta', 'cyan', 'black'][i % 6]
            key = names[uuid]

            employment = data[uuid]['employment'][-1]
            runtime = len(data[uuid]['production']) * 0.1
            production = sum(data[uuid]['production']) / runtime
            cost_hiring = sum(data[uuid]['cost_hiring']) / runtime
            cost_salary = sum(data[uuid]['cost_salary']) / runtime
            interv_private = sum(data[uuid]['interv_private']) / runtime
            interv_public = sum(data[uuid]['interv_public']) / runtime

            cost = cost_hiring+cost_salary+interv_private

            score = employment
            if production < cost:
                score = 0
            values = [dict(x=0, y=score), dict(x=1, y=employment)
                      ]
            bar.append(dict(values=values, key=key, color=color))


        return json.dumps(dict(bar=bar))

    def get_name(self, uuid):
        if isinstance(uuid, uuid_package.UUID):
            uuid = str(uuid)
        if uuid not in names:
            name = 'User %d' % (len(names) + 1)
            names[uuid] = name
        else:
            name = names[uuid]
        return name


    def swi_play(self, uuid=None):
        if uuid is None:
            uuid = str(uuid_package.uuid4())
        if not actions.has_key(uuid):
            actions[uuid] = []
            action_texts[uuid] = []
            seeds[uuid] = uuid_package.UUID(uuid).int & 0x7fffffff
        if self.user is None:
            return self.create_login_form()
        html = pkgutil.get_data('employ_game', 'templates/play.html')
        name = self.get_name(uuid)
        return html % dict(uuid=uuid, name=name, seed=seeds[uuid])

    def swi_set_name(self, uuid, name):
        names[uuid] = name

    def run_game(self, u):
        seed = seeds[u]
        acts = actions[u]

        return model.run(seed, *acts)



    def swi_play_json(self, uuid, action, action_text, seed=None):
        maximum = 10
        substeps = 10
        name = self.get_name(uuid)
        if action == 'init':
            actions[uuid] = []
            action_texts[uuid] = []
        if seed is not None:
            seeds[uuid] = int(seed)

        if action == 'undo':
            if len(actions[uuid]) > 1:
                del actions[uuid][-1]
                del action_texts[uuid][-1]
        elif len(actions[uuid]) >= maximum:
            pass
        else:
            actions[uuid].append(action)
            action_texts[uuid].append(action_text)

        data = self.run_game(uuid)

        time = []
        for i in range(2):
            color = ['blue', 'green', 'red', 'magenta', 'cyan', 'black'][i % 6]
            key = ['employment', 'highschool'][i]
            values = []
            for j in range(len(data[key])):
                values.append(dict(x=float(j)/substeps, y=data[key][j]))
            values.append(dict(x=maximum, y=None))
            time.append(dict(values=values, key=key, color=color, area=False))

        race = []
        race_pie = []
        for k in sorted(data.keys()):
            if k.startswith('employment_'):
                key = k[11:]
                color = ['blue', 'red', 'black', 'magenta', 'cyan', 'green'][len(race) % 6]
                values = []
                for j in range(len(data[k])):
                    values.append(dict(x=float(j)/substeps, y=data[k][j]))
                values.append(dict(x=maximum, y=None))
                race.append(dict(values=values, key=key, color=color, area=False))

                p = data['proportion_%s' % key]
                race_pie.append(dict(label=key, value=p[-1], color=color))


        grid = data['grid']

        money = []
        runtime = len(data['production']) * 0.1
        production = sum(data['production']) / runtime
        cost_hiring = sum(data['cost_hiring']) / runtime
        cost_salary = sum(data['cost_salary']) / runtime
        interv_private = sum(data['interv_private']) / runtime
        interv_public = sum(data['interv_public']) / runtime
        money.append(dict(key='production', values=[{'x':0, 'y':production}, {'x':1, 'y':0}, {'x':2, 'y':0}]))
        money.append(dict(key='hiring', values=[{'x':0, 'y':0}, {'x':1, 'y':cost_hiring}, {'x':2, 'y':0}]))
        money.append(dict(key='salary', values=[{'x':0, 'y':0}, {'x':1, 'y':cost_salary}, {'x':2, 'y':0}]))
        money.append(dict(key='intervention', values=[{'x':0, 'y':0}, {'x':1, 'y':interv_private}, {'x':2, 'y':interv_public}]))

        a = action_texts[uuid][1:]
        a = ['%d: %s' % (i+1,x) for i,x in enumerate(a)]
        a = '<br/>'.join(a)

        return json.dumps(dict(time=time, race=race, race_pie=race_pie, grid=grid, money=money, actions=a))


    def create_login_form(self):
        message = "Enter the password:"
        if self.attemptedLogin:
            message = "Invalid password"
        return """<form action="/" method="POST">%s<br/>
        <input type=hidden name=swi_id value="" />
        <input type=password name=swi_pwd>
        </form>""" % message

