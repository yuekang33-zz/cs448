/**
 * Reference : joonkukang on 2014. 1. 19..
 */
// Reference : 'Programming Collective Intellignece' by Toby Segaran.
/*
var data =[['slashdot','USA','yes',18],
           ['google','France','yes',23],
           ['digg','USA','yes',24],
           ['kiwitobes','France','yes',23],
           ['google','UK','no',21],
           ['(direct)','New Zealand','no',12],
           ['(direct)','UK','no',21],
           ['google','USA','no',24],
           ['slashdot','France','yes',19],
           ['digg','USA','no',18,],
           ['google','UK','no',18,],
           ['kiwitobes','UK','no',19],
           ['digg','New Zealand','yes',12],
           ['slashdot','UK','no',21],
           ['google','UK','yes',18],
           ['kiwitobes','France','yes',19]];
var result = ['None','Premium','Basic','Basic','Premium','None','Basic','Premium','None','None','None','None','Basic','None','Basic','Basic'];
*/
var _ = require('lodash');
var training_data = require('./iris.json')


var class_name = Object.keys(training_data[0])[Object.keys(training_data[0]).length-1];
var features = Object.keys(training_data[0]).slice(0,Object.keys(training_data[0]).length-1)

var data=[];
for(var k in training_data){
	data.push([]);
	for(var f in features){
		data[k].push(training_data[k][features[f]])
	}
}

var result = _.pluck(training_data,class_name);

//var ml = require('./ML/lib/machine_learning');
var decisionTree = require('./ML/lib/DecisionTree');
//var dt = new ml.DecisionTree({
var dt = new decisionTree({
    data : data,
    result : result,
	feature : features
});

dt.build();

// dt.print();

console.log("Classify : ", dt.classify([data[0]]));

//dt.prune(1.0); // 1.0 : mingain.
dt.print();

