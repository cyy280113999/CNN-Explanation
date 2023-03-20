# Heatmap Explanation-Visualization&Evaluation
DNN Explanation Visualization and Evaluation

## Content
Load DataSet By Click

Heatmap Visualization

Mask Method

Heatmap Evaluation

## Notice
Only For Vgg16

## Demo
![demo](https://github.com/cyy280113999/Explanation-Visualization/blob/main/demo.png)

## Usage

### Visualization:

1. store customized Heatmap Generator Script in '/methods/'

       # Heatmap Generator must be called by 'x, y' , return heatmap

2. import & add prompt in 'ExpVis.py-ExplainMethodSelector-methods'

        # the method prompt interface, all methods must follow this:
        # the method prompt can be called twice
        # 1. the method first accept "model" parameter, create a callable function "_m = m(model)"
        # 2. the heatmap generated by secondly calling "hm = _m(x,yc)"
        # easily coding by lambda model: lambda x,y: function

3. start 'ExpVis.py' , choose dataset, model, heatmap method. A Screen will show

### Evaluation 

1. import & add prompt in 'ExpEval.py-EvaluatorSetter-heatmap_methods'

2. store customized Heatmap Evaluator in '/Evaluators/' 

       # Evaluator must be accept dataset, model, heatmap_method
       # Evaluator will get scores in every sample evaluating.
       # Evaluator must generate output string to save.

3. in 'ExpEval.py' , set all evaluation settings in 'main'

4. run, Evaluation Results are stored in '/datas/'

5. you can find some compatible Analyser in '/EvalAnalysers/', code yourself analyser

6. we provide 'Prob Change','Maximal Patch', 'Point Game' Evaluators.

