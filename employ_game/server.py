import os.path
import json
import random
import uuid

import pkgutil
import employ_game

actions = {}
seeds = {}

class Server(employ_game.swi.SimpleWebInterface):
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
        else:
            raise Exception('unknown extenstion for %s' % fn)

        data = pkgutil.get_data('employ_game', fn)
        return (mimetype, data)

    def swi_favicon_ico(self):
        icon = pkgutil.get_data('employ_game', 'static/favicon.ico')
        return ('image/ico', icon)

    def swi(self):
        if self.user is None:
            return self.create_login_form()
        html = pkgutil.get_data('employ_game', 'templates/index.html')
        return html % dict(uuid=uuid.uuid4())
        
    def swi_overview(self):
        if self.user is None:
            return self.create_login_form()
        html = pkgutil.get_data('employ_game', 'templates/overview.html')
        return html
        
    def swi_overview_json(self):
        maximum = 10
        substeps = 20
        
        uuids = [k for k in actions.keys() if len(actions[k])>1]
        
        data = {k: self.run_game(k) for k in uuids}
        
        time = []
        for i, uuid in enumerate(uuids):
            color = ['blue', 'red', 'green', 'magenta', 'cyan', 'black'][i % 6]
            key = uuid
            values = []
            for j in range(len(data[uuid]['employment'])):
                values.append(dict(x=float(j)/substeps, y=data[uuid]['employment'][j]))
            values.append(dict(x=maximum, y=None))
            time.append(dict(values=values, key=key, color=color, area=False))    
            
        bar = []
        for i, uuid in enumerate(uuids):
            color = ['blue', 'red', 'green', 'magenta', 'cyan', 'black'][i % 6]
            key = uuid
            values = [dict(x=0, y=data[uuid]['employment'][-1]), 
                      dict(x=1, y=data[uuid]['cost'][-1]), 
                      dict(x=2, y=data[uuid]['employment'][-1] / data[uuid]['cost'][-1])]
            bar.append(dict(values=values, key=key, color=color))    
            
    
        return json.dumps(dict(time=time, bar=bar))

    def swi_play(self, uuid):
        if self.user is None:
            return self.create_login_form()
        html = pkgutil.get_data('employ_game', 'templates/play.html')
        return html % dict(uuid=uuid)
        
    def run_game(self, u):
        substeps = 20
        steps = len(actions[u]) * substeps
        
        # TODO: add command line argument to set this seed globally
        random.seed(uuid.UUID(u).int)
        
        employment = [0]
        cost = [0]
        for t in range(steps):
            intervention = actions[u][t/substeps]
            
            if intervention == 'hire people':
                de = 10
                dc = 10
            elif intervention == 'downsize':
                de = -10
                dc = -10
            else:
                de = 0
                dc = 0
            
            employment.append(random.gauss(employment[-1]+de, 5))
            cost.append(random.gauss(cost[-1]+dc, 5))
        return dict(employment=employment, cost=cost)
    
    
        
    def swi_play_json(self, uuid, action):
        maximum = 10
        substeps = 20
        if not actions.has_key(uuid) or action=='init':
            actions[uuid] = []
            
        if action == 'undo':
            if len(actions[uuid]) > 1:
                del actions[uuid][-1]
        elif len(actions[uuid]) >= maximum:
            pass
        else:
            actions[uuid].append(action)
    
        data = self.run_game(uuid)
        
        time = []
        for i in range(2):
            color = ['blue', 'red', 'green', 'magenta', 'cyan', 'black'][i % 6]
            key = ['employment', 'cost'][i]
            values = []
            for j in range(len(data[key])):
                values.append(dict(x=float(j)/substeps, y=data[key][j]))
            values.append(dict(x=maximum, y=None))
            time.append(dict(values=values, key=key, color=color, area=False))    
        return json.dumps(time)
        

    def create_login_form(self):
        message = "Enter the password:"
        if self.attemptedLogin:
            message = "Invalid password"
        return """<form action="/" method="POST">%s<br/>
        <input type=hidden name=swi_id value="" />
        <input type=password name=swi_pwd>
        </form>""" % message

