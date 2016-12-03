/**
 * Reference : joonkukang's github on machine learning
 */
var utils = require('./utils');
var math = utils.math;
function DecisionNode(options) {
    var self = this;
    self.col = (typeof options['col'] === 'undefined') ? -1 : options['col'];
    self.value = options['value'];
    self.results = options['results'];
	self.name = options['name'];
	self.children = options['children'],
	self.pNode = options['pNode'],//parent node, naming as 'parent' cause naming conflict
	self.partObj = options['partObj'];
}
DecisionNode.prototype.print =  function() {
    var self = this;
    printTree(self,'');
}

let DecisionTree = module.exports = function (options) {
    var self = this;
    self.data = options['data'];
    self.result = options['result'];
	self.feature = options['feature'];
}

DecisionTree.prototype.build = function(options) {
    var self = this;
    var rows = [];
    var i;
    for(i=0; i<self.data.length; i++) {
		//console.log('testtest',self.data[i]);
        rows.push(self.data[i]);
        rows[i].push(self.result[i]);
    }
    self.tree = buildTree('null',rows,entropy,self.feature);
    return self.tree;
};

DecisionTree.prototype.print = function() {

    var self = this;
    printTree(self.tree,'');
}

DecisionTree.prototype.classify = function(observation) {
    var self = this;
    return classify(observation,self.tree);
}

DecisionTree.prototype.prune = function(mingain){
    var self = this;
    prune(self.tree,mingain);
}

DecisionTree.prototype.getTree = function() {
    return this.tree;
}

function prune(tree,mingain) {
    if(typeof tree.children[0].results === 'undefined')
        prune(tree.children[0],mingain);
    if(typeof tree.children[1].results === 'undefined')
        prune(tree.children[1],mingain);

    if(typeof tree.children[0].results !== 'undefined' && typeof tree.children[1].results !== 'undefined') {
        var tb = [], fb = [], tbfb = [];
        Object.keys(tree.children[0].results).forEach(function(key) {
            for(var i=0; i<tree.children[0].results[key] ; i++) {
                tb.push([key]);
                tbfb.push([key]);
            }
        });
        Object.keys(tree.children[1].results).forEach(function(key) {
            for(var i=0; i<tree.children[1].results[key] ; i++) {
                fb.push([key]);
                tbfb.push([key]);
            }
        });
        var p = 1.*tb.length / tbfb.length;
        var delta = entropy(tbfb) - p * entropy(tb) - (1-p) * entropy(fb);
        if(delta < mingain) {
            // pruning
            tree.children[0] = undefined;
            tree.children[1] = undefined;
            tree.results = uniqueCounts(tbfb);
        }
    }
}

function classify(observaton,tree) {
    if(typeof tree.results !== 'undefined')
        return tree.results;
    else {
        var v = observaton[tree.col];
        var branch;
        if(utils.isNumber(v)) {
            if(v>=tree.value) branch = tree.children[0];
            else branch = tree.children[1];
        } else {
            if(v === tree.value) branch = tree.children[0];
            else branch = tree.children[1];
        }
        return classify(observaton,branch);
    }
}

function printTree(tree,indent) {
	console.log(tree);
	var fs = require('fs');
	fs.writeFile("test_iris.txt", JSON.stringify(tree), function(err) {
		if(err) {
			return console.log(err);
		}
	});
}

function buildTree(pNode,rows,scoref,feature) {
    if(rows.length == 0) return new DecisionNode();
    var currentScore = scoref(rows);
    var bestGain = 0.0, bestCriteria, bestSets;
    var columnCount = rows[0].length - 1;
    var col, i;
    for(col=0; col<columnCount; col++) {
        var columnValues = {};
        for(i=0; i<rows.length; i++) {
            columnValues[rows[i][col]] = 1;
        }
        var values = Object.keys(columnValues);
        for(i=0; i<values.length; i++) {
            var sets = divideSet(rows,col,values[i]);
            var p = 1.*sets[0].length / rows.length;
            var gain = currentScore - p*scoref(sets[0]) - (1-p)*scoref(sets[1]);
            if(gain > bestGain && sets[0].length > 0 && sets[1].length > 0) {
                bestGain = gain;
                bestCriteria = [col,values[i]];
                bestSets = sets;
            }
        }
    }
    if(bestGain > 0) {
		var nodeName;
		if(utils.isNumber(bestCriteria[1])){
			nodeName = feature[bestCriteria[0]]+'>='+bestCriteria[1];
		}
		else{
			nodeName = feature[bestCriteria[0]]+'=='+bestCriteria[1];
		}
		var trueBranch = buildTree(nodeName,bestSets[0],scoref,feature);
        var falseBranch = buildTree(nodeName,bestSets[1],scoref,feature);
		var curPartition = uniqueCounts(rows);
        return new DecisionNode({
            col : bestCriteria[0],
            value : bestCriteria[1],
			name : nodeName,
			children:[trueBranch,falseBranch],
			pNode : pNode,
			partObj : curPartition
        });
    } else {
		//leaf node, no name, result is the majority
		var curPartition = uniqueCounts(rows);
        return new DecisionNode({
			name : ' ',
			pNode : pNode,
			partObj : curPartition,
            results : majClass(curPartition)
        });
    }
}

function entropy(rows) {
    var log2 = function(x) {return Math.log(x)/Math.log(2);};
    var results = uniqueCounts(rows);
    var ent = 0.0;
    var keys = Object.keys(results);
    var i;
    for(i=0; i<keys.length; i++) {
        var p = 1.*results[keys[i]]/rows.length;
        ent -= 1.*p*log2(p);
    }
    return ent;
}

function majClass(curPartition){
	var majClass;
	var maxCount = 0;
	for(var k in curPartition){
		if(curPartition[k]>=maxCount){
			maxCount = curPartition[k];
			majClass = k;
		}
	}
	return majClass;
}

function uniqueCounts(rows) {
    var results = {};
    var i;
    for(i=0; i<rows.length; i++) {
        var r = rows[i][rows[i].length-1];
        if(typeof results[r] === 'undefined')
            results[r] = 0;
        results[r]++;
    }
    return results;
}
function divideSet(rows,column,value) {
    var splitFunction;

    if(utils.isNumber(value))
        splitFunction = function(row) {return row[column] >= value;};
    else
        splitFunction = function(row) {return row[column] === value;};
    var set1 = [], set2 = [];
    var i;
    for(i=0; i<rows.length; i++) {
        if(splitFunction(rows[i]))
            set1.push(rows[i]);
        else
            set2.push(rows[i]);
    }
    return [set1,set2];
}
