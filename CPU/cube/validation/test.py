#this quick and dirty script will read from the file 'data'
#and run 2 prorgams (ours.out, which is the one we want to test)
#and twist.out,which is the ground truth that we are testing on


import subprocess, shlex;

map = {
    'U'  : '0',
    'U2' : '1',
    'U\'' : '2',
    'F'  : '3',
    'F2' : '4',
    'F\'' : '5',
    'R'  : '6',
    'R2' : '7',
    'R\'' : '8',
    'D'  : '9',
    'D2' : '10',
    'D\'' : '11',	
    'B'  : '12',
    'B2' : '13',
    'B\'' : '14',
    'L'  : '15',
    'L2' : '16',
    'L\'' : '17'
 

}

f = open('data','r')

nsuccess = 0;
nfail = 0;
sm_success = 0;
sm_fail = 0

i=1;

for line in f:
	line = line.strip('\n')	
	oursstr= line.split(' ');
	ours = []
	for our in oursstr:
		ours.append(map[our])	



	ours = ["bin/ours.out"] + ours;
	twist = ["bin/twist.out"] + line.split(' ');
	sm = ["bin/sm_test.out"] + ours;

	proc_ours = subprocess.Popen(ours, stdout=subprocess.PIPE);
	proc_twist = subprocess.Popen( twist, stdout=subprocess.PIPE);
	proc_sm = subprocess.Popen( sm, stdout=subprocess.PIPE);



	output_ours  = proc_ours.stdout.read()
	output_twist = proc_twist.stdout.read();
	output_sm = proc_sm.stdout.read();


	print  str(i) + ' )'
	
	print '    a) Testing move performance';
	if (output_twist == output_ours):
		print "    ....SUCCESS!"
		nsuccess += 1;

	else:
		print "    ....FAILED!!!"
		nfail += 1;
	
	
	print '    b) testing singmaster parsing'
        
	if (output_sm == "1"):
		print '    ....SUCCESS!'
		sm_success += 1
	else:
		print '    ....FAILED!!!!'
		sm_fail += 1
	print '\n'	 
	i += 1;



	


print '============================='
print 'move results'
print 'nsuccess: ' + str(nsuccess)
print 'nfail: ' + str(nfail)

print '================================'
print 'singmaster parsing results'
print 'parsing success: '  + str(sm_success)
print 'parsing failed: ' + str(sm_fail)
