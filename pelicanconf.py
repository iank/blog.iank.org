#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = u'Ian Kilgore'
SITENAME = u'blog'
SITEURL = 'http://blog.iank.org'

PATH = 'content'

TIMEZONE = 'America/New_York'

DEFAULT_LANG = u'en'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None

# Blogroll
LINKS = (('Blog', 'http://blog.iank.org/'),
         ('Contact', 'http://blog.iank.org/pages/contact.html'),
         ('Home', 'http://iank.org/'),
#         ('Résumé', 'http://iank.org/kilgore_resume.pdf'),
         ('Projects', 'http://iank.org/projects.html'),)

# Social widget
#SOCIAL = (('You can add links in your config file', '#'),
#          ('Another social link', '#'),)

DEFAULT_PAGINATION = 10

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True

THEME = "/home/ian/blog/theme/"
MD_EXTENSIONS = ['codehilite', 'extra', 'footnotes']
STATIC_PATHS = ['images']
