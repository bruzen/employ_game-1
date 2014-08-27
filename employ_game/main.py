import employ_game.swi
import employ_game.server

def main():
    port = 80
    addr = ''
    employ_game.swi.browser(port)
    employ_game.swi.start(employ_game.server.Server, port, addr=addr)

if __name__ == '__main__':
    main()
