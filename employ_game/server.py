import os.path
import json
import random

import pkgutil
import employ_game

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
        return html
        
    def swi_overview(self):
        if self.user is None:
            return self.create_login_form()
        html = pkgutil.get_data('employ_game', 'templates/overview.html')
        return html
        
    def swi_overview_json(self):
        maximum = 10
        users = 5
        substeps = 20
        steps = random.randrange(1, maximum) * substeps
        #steps = 3 * substeps
        
        time = []
        for i in range(users):
            color = ['blue', 'red', 'green', 'magenta', 'cyan', 'black'][i % 6]
            key = 'User%d' % i
            values = [dict(x=0, y=0)]
            for j in range(1, steps):
                y2 = values[-1]['y'] + random.gauss(0, 5)
                values.append(dict(x=float(j)/substeps, y=y2))
            values.append(dict(x=maximum, y=None))
            time.append(dict(values=values, key=key, color=color, area=False))    
            
        bar = []
        for i in range(users):
            color = ['blue', 'red', 'green', 'magenta', 'cyan', 'black'][i % 6]
            key = 'User%d' % i
            values = [dict(x=0, y=random.gauss(1, 0.5)), dict(x=1, y=random.gauss(2, 0.7)), dict(x=2, y=random.gauss(0, 0.3))]
            bar.append(dict(values=values, key=key, color=color))    
            
    
        return json.dumps(dict(time=time, bar=bar))

    def swi_play(self):
        if self.user is None:
            return self.create_login_form()
        html = pkgutil.get_data('employ_game', 'templates/play.html')
        return html
        

    def create_login_form(self):
        message = "Enter the password:"
        if self.attemptedLogin:
            message = "Invalid password"
        return """<form action="/" method="POST">%s<br/>
        <input type=hidden name=swi_id value="" />
        <input type=password name=swi_pwd>
        </form>""" % message

