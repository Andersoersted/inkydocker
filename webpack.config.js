const path = require('path');

module.exports = {
  entry: './assets/js/main.js', // Create this file to initialize FullCalendar, etc.
  output: {
    filename: 'bundle.js',
    path: path.resolve(__dirname, 'static', 'js')
  },
  module: {
    rules: [
      {
        test: /\.css$/i,
        use: ['style-loader', 'css-loader'],
      },
    ],
  },
};
