function [x,f,exitflag,output] = minFunc(funObj,x0,options,X,Y)
% Inputs:
  %   funObj - is a function handle
  %   x0 - is a starting vector
  %   options - is a struct containing parameters (defaults are used for non-existent or blank fields)
  %   varargin{:} - all other arguments are passed as additional arguments to funObj
  %
  % Outputs:
  %   x is the minimum value found
  %   f is the function value at the minimum found
  %   exitflag returns an exit condition
  %   output returns a structure with other information

  fun = @(theta) [f, g] = funObj(theta, X, Y);

  [x, f, exitflag, output] = fminunc(fun, x0, options);
endfunction
