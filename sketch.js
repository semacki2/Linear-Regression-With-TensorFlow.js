let xs = [];
let ys = [];
let m;
let b;

//create an optimizer to minimize our loss
//we are using stochasitc gradient descent
//slowling tweaking the variable of our linear regression function until it accuratly
//predicts y values. 
const learningRate = 0.2;
const optimizer = tf.train.sgd(learningRate);

function setup() {
  createCanvas(400, 400);
  background(0);

    createP(
    "Click on the canvas to add points. </br> Refresh the page to start over."
  );
  
  //these need to be variables because we will be changing them
  m = tf.variable(tf.scalar(random(1)));
  b = tf.variable(tf.scalar(random(1)));


}

function draw() {
  //if there are values in our arrays
  if (xs.length > 1 && ys.length > 1) {
    //use tf.tidy() to prevent memory leaks
    tf.tidy(() => {
      //create a tensor out of the outputs
      const tfys = tf.tensor1d(ys);

      //train the model
      //adjust m and b to minimize the "loss" function
      //the minimize function automatically trains by adjusting variables.
      //the only variables we created were m and b.
      //we could be explicit and define a varList of variables to adjust.
      optimizer.minimize(() =>
        loss(predict(xs), tfys)
      );
    });


  }

  background(0);

  stroke(255);
  strokeWeight(8);



  for (let i = 0; i < xs.length; i++) {
    let px = map(xs[i], 0, 1, 0, width);
    let py = map(ys[i], 1, 0, 0, height);
    point(px, py);
  }

  //draw the line if there are 2 or more points
  if (xs.length > 1 && ys.length > 1) {
    //draw the line as defined by current values of m and b
    //y = mx + b

    //lxs = line x values
    const lxs = [0, 1];

    //lys = line y values
    const lys = tf.tidy(() => predict(lxs));

    //convert it from a tensor back to an array
    let lineY = lys.dataSync();

    let x1 = map(lxs[0], 0, 1, 0, width);
    let y1 = map(lineY[0], 0, 1, height, 0);
    let x2 = map(lxs[1], 0, 1, 0, width);
    let y2 = map(lineY[1], 0, 1, height, 0);

    stroke(255);
    strokeWeight(1);
    line(x1, y1, x2, y2);
    
    lys.dispose();
  }

  //print(tf.memory().numTensors);
}

function mousePressed() {

  let x = map(mouseX, 0, width, 0, 1);
  let y = map(mouseY, 0, height, 1, 0);

  xs.push(x);
  ys.push(y);
}

//function that takes in an array of x values and returns an array of y values.
function predict(xs) {
  //create a tensor out of the inputs
  const tfxs = tf.tensor1d(xs);

  //y = mx + b
  //in this case, m and b are global variables.
  const ys = tfxs.mul(m).add(b);

  return ys;
}

//loss function. tells us how wrong our model was.
//predictions are the y values from the predict function
//labels are the "true" y values stored in the ys array
//finds the distance between each prediction and the "true" value.
//squares the difference.
//returns the mean of all the squared errors.
function loss(predictions, ys) {
  return predictions.sub(ys).square().mean();
}