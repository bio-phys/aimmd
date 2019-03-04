"""
Default logging configuration dictionary for ops/arcd.

Load the dict and modify it before loading if you want to e.g. change
logfilenames. This is the same as the OpenPathSampling logconf file,
except it is a config dict, it is therefore easier to change single values
like logfilenames.
"""

LOGCONFIG = {'version': 1,
             'disable_existing_loggers': False,
             'formatters': {'standardFormatter': {
                               'class': 'logging.Formatter',
                               'format': '(%(levelname)s)%(name)s: %(message)s'
                                                  },
                            'msgOnly': {
                                  'class': 'logging.Formatter',
                                  'format': '%(message)s'
                                        }
                            },
             'handlers': {'stdout': {
                                     'class': 'logging.StreamHandler',
                                     'level': 'NOTSET',
                                     # 'stream': 'sys.stdout',
                                     'formatter': 'msgOnly'
                                      },
                          'warnout': {
                                      'class': 'logging.StreamHandler',
                                      'level': 'WARN',
                                      'formatter': 'standardFormatter'
                                      },
                          'initf': {
                                    'class': 'logging.FileHandler',
                                    'level': 'INFO',
                                    'mode': 'w',
                                    'filename': 'initialization.log',
                                    'formatter': 'standardFormatter'
                                    },
                          'stdf': {
                                   'class': 'logging.FileHandler',
                                   'level': 'INFO',
                                   'mode': 'w',
                                   'filename': 'simulation.log',
                                   'formatter': 'standardFormatter'
                                   }
                          },
             # need to name root logger '' to update the default root logger
             # at least inside an ipython notebook
             'loggers': {'': {
                              'level': 'INFO',
                              'handlers': ['stdf', 'warnout']
                              },
                         'openpathsampling.initialization': {
                                'level': 'INFO',
                                'handlers': ['initf'],
                                'qualname': 'openpathsampling.initialization',
                                'propagate': 0
                                                             }
                         }
             }
