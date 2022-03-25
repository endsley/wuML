#!/usr/bin/env python

import datetime

#	convert a date like '2/10/2021', 'Feb/10/2021' to an integer
def date_string_to_number(date_string, pattern='%m/%d/%Y'):
	#note: use dateObj as , dateObj.year, dateObj.month, dateObj.day

	try:
		dateObj = datetime.datetime.strptime(date_string, pattern)
	except:
		pattern='%m/%d/%y'		# This will handle 2/12/19 instead of 2/12/2019
		dateObj = datetime.datetime.strptime(date_string, pattern)

	return dateObj.timestamp()


