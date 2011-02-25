TESTRUNNER=nosetests --with-coverage --with-doctest --cover-package=psychic

.PHONY: all test

all: test 

test:
	$(TESTRUNNER) psychic
