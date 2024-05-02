
y=2019




def era(s):
    l = {'2009-2013':list(range(2009,2014)),'2014-2017':list(range(2014,2018)),'2017-2021':list(range(2018,2022))}
    for key, years_list in l.items():
        if s in years_list:
            return key
        
        
