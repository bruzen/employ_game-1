import employ_game.swi
import employ_game.server


import sys
def main():
    port = int(sys.argv[1]) if len(sys.argv)>1 else 8080
    addr = ''
    employ_game.swi.browser(port)
    employ_game.swi.start(employ_game.server.Server, port, addr=addr)

if __name__ == '__main__':
    main()
