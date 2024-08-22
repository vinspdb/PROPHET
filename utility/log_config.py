dict_log = {}
dict_log['logboek'] = {'event_attr': ["activity", "timesincecasestart", "resource"], 'trace_attr_num':[], 'trace_attr_cat': ['Diagnose','Leeftijd bij opname','Geslacht','Ontslagbestemming'],
                         'relation': [ ('activity', 'follow', 'activity'),
                                       ('resource', 'follow', 'resource'),
                                       ('timesincecasestart', 'follow', 'timesincecasestart'),

                                       ('activity', 'done', 'resource'),
                                       ('activity', 'executed', 'timesincecasestart'),

                                       ('resource', 'used', 'timesincecasestart'),

                                       ('activity', 'has_ta', 'trace_att'),
                                       ('resource', 'has_ta', 'trace_att'),
                                       ('timesincecasestart', 'has_ta', 'trace_att'),

                                       ]}


dict_log['sp2020'] = {'event_attr': ["activity", "timesincecasestart"], 'trace_attr_num':['REPAIR_IN_TIME_5D'], 'trace_attr_cat': ['DEVICETYPE','SERVICEPOINT'],
                         'relation': [ ('activity', 'follow', 'activity'),
                                       ('timesincecasestart', 'follow', 'timesincecasestart'),
                                       ('activity', 'executed', 'timesincecasestart'),
                                       ('activity', 'has_ta', 'trace_att'),

                                       ('timesincecasestart', 'has_ta', 'trace_att')

                                       ]}

dict_log['helpdesk2'] = {'event_attr': ["activity", "timesincecasestart"], 'trace_attr_num':[], 'trace_attr_cat': [],
                         'relation': [ ('activity', 'follow', 'activity'),
                                       ('timesincecasestart', 'follow', 'timesincecasestart'),
                                       ('activity', 'executed', 'timesincecasestart'),
                                       ]

                         }
dict_log['openproblem'] = {'event_attr': ["activity", "resource", "timesincecasestart", "orggroup", "orgrole","resourcecountry"], 'trace_attr_num':[], 'trace_attr_cat': ["impact","product","orgcountry"],
                               'relation':[
                                            ('activity', 'follow', 'activity'),
                                            ('resource', 'follow', 'resource'),
                                            ('resourcecountry', 'follow', 'resourcecountry'),
                                            ('orggroup', 'follow', 'orggroup'),
                                            ('orgrole', 'follow', 'orgrole'),
                                            ('timesincecasestart', 'follow', 'timesincecasestart'),

                                            ('activity', 'done', 'resource'),
                                            ('activity', 'executed', 'timesincecasestart'),
                                            ('activity', 'has', 'orggroup'),
                                            ('activity', 'has', 'orgrole'),
                                            ('activity', 'has', 'resourcecountry'),

                                            ('resource', 'used', 'timesincecasestart'),
                                            ('resource', 'has', 'orggroup'),
                                            ('resource', 'has', 'orgrole'),
                                            ('resource', 'has', 'resourcecountry'),

                                            ('timesincecasestart', 'has', 'orggroup'),
                                            ('timesincecasestart', 'has', 'orgrole'),
                                            ('timesincecasestart', 'has', 'resourcecountry'),

                                            ('orggroup', 'has', 'orgrole'),
                                            ('orggroup', 'has', 'resourcecountry'),

                                            ('orgrole', 'has', 'resourcecountry'),

                                            ('activity', 'has_ta', 'trace_att'),
                                            ('resource', 'has_ta', 'trace_att'),
                                            ('resourcecountry', 'has_ta', 'trace_att'),
                                            ('orggroup', 'has_ta', 'trace_att'),
                                            ('orgrole', 'has_ta', 'trace_att'),
                                            ('timesincecasestart', 'has_ta', 'trace_att')
                                           ]}


dict_log['bpi12w_complete'] = {'event_attr': ["activity", "resource", "timesincecasestart"], 'trace_attr_cat':[],'trace_attr_num': ['amount'],
                          'relation':[
                                      ('activity', 'follow', 'activity'),
                                      ('resource', 'follow', 'resource'),
                                      ('timesincecasestart', 'follow', 'timesincecasestart'),

                                      ('activity', 'done', 'resource'),
                                      ('activity', 'executed', 'timesincecasestart'),

                                      ('resource', 'used', 'timesincecasestart'),

                                      ('activity', 'has_ta', 'trace_att'),
                                      ('resource', 'has_ta', 'trace_att'),
                                      ('timesincecasestart', 'has_ta', 'trace_att')]}

dict_log['bpi12a'] = {'event_attr': ["activity", "resource", "timesincecasestart"], 'trace_attr_cat':[],'trace_attr_num': ['amount'],
                          'relation':[
                                      ('activity', 'follow', 'activity'),
                                      ('resource', 'follow', 'resource'),
                                      ('timesincecasestart', 'follow', 'timesincecasestart'),

                                      ('activity', 'performed', 'resource'),
                                      ('activity', 'executed', 'timesincecasestart'),
                                      ('resource', 'used', 'timesincecasestart'),

                                      ('activity', 'has_ta', 'trace_att'),
                                      ('resource', 'has_ta', 'trace_att'),
                                      ('timesincecasestart', 'has_ta', 'trace_att')
                                      ]}

dict_log['bpic2017_o2'] = {'event_attr': ["activity", "resource", "action", "timesincecasestart"], 'trace_attr_cat':[],'trace_attr_num': ["MonthlyCost", "CreditScore", "FirstWithdrawalAmount", "OfferedAmount", "NumberOfTerms"],
                          'relation':[
                                      ('activity', 'follow', 'activity'),
                                      ('resource', 'follow', 'resource'),
                                      ('action', 'follow', 'action'),
                                      ('timesincecasestart', 'follow', 'timesincecasestart'),

                                      ('activity', 'done', 'resource'),
                                      ('activity', 'executed', 'resource'),
                                      ('activity', 'has', 'action'),

                                      ('resource', 'used', 'timesincecasestart'),
                                      ('resource', 'has', 'action'),

                                      ('timesincecasestart', 'has', 'action'),

                                      ('activity', 'has_ta', 'trace_att'),
                                      ('resource', 'has_ta', 'trace_att'),
                                      ('action', 'has_ta', 'trace_att'),
                                      ('timesincecasestart', 'has_ta', 'trace_att')]}


dict_log['bpic20202'] = {'event_attr': ["activity", "resource", "Role", "timesincecasestart"], 'trace_attr_num':[],'trace_attr_cat': ["Org", "Project", "Task"],
                        'relation':[
                                    ('activity', 'follow', 'activity'),
                                    ('resource', 'follow', 'resource'),
                                    ('Role', 'follow', 'Role'),
                                    ('timesincecasestart', 'follow', 'timesincecasestart'),

                                    ('activity', 'done', 'resource'),
                                    ('activity', 'executed', 'timesincecasestart'),
                                    ('activity', 'has', 'Role'),

                                    ('resource','used','timesincecasestart'),
                                    ('resource', 'has', 'Role'),

                                    ('timesincecasestart', 'has', 'Role'),

                                    ('activity', 'has_ta', 'trace_att'),
                                    ('resource', 'has_ta', 'trace_att'),
                                    ('Role', 'has_ta', 'trace_att'),
                                    ('timesincecasestart', 'has_ta', 'trace_att')
                                    ]}



dict_log['invoice2'] = {'event_attr': ["activity", "resource", "timesincecasestart"], 'trace_attr_num': ['InvoiceTotalAmountWithoutVAT'], 'trace_attr_cat': ['CostCenter.Code','Supplier.City','Supplier.Name','Supplier.State'],
                               'relation':[
                                            ('activity', 'follow', 'activity'),
                                            ('resource', 'follow', 'resource'),
                                            ('timesincecasestart', 'follow', 'timesincecasestart'),

                                            ('activity', 'done', 'resource'),
                                            ('activity', 'executed', 'timesincecasestart'),

                                            ('resource','used','timesincecasestart'),

                                            ('activity', 'has_ta', 'trace_att'),
                                            ('resource', 'has_ta', 'trace_att'),
                                            ('timesincecasestart', 'has_ta', 'trace_att')]}