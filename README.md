employ_game
===========

Prototype youth employment game


Installation
============

This is an app bundled as a python package. To install, use
```
$ python setup.py install
```

If you are working on it, try
```
$ python setup.py develop --user
```

Which should add `employ_game` to a `.pth` file in a folder that is on your `sys.path`, allowing you to import it.
This is important because it uses pkgutil to find static web content and such, a module which only works on installed packages.
