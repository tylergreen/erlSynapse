%_____________________________________________
% Neural Net Implementation written in Erlang
% mostly copied from http://trapexit.org/Erlang_And_Neural_Networks
% original author Wilhelm

% slightly modified by Tyler Green 2010

-module(neuralnet).
-import(lists,[sum/1, zipwith/3]).
-compile(export_all).



dot_prod(Xs, Ys) ->
    sum(zipwith(fun(X,Y) -> X * Y end,  Xs, Ys)).

sigmoid(X) ->
    1 / (1 + math:exp(-X)).

sigmoid_deriv(X) ->
    math:exp(-X) / (1 + math:exp(-2 * X)).

% where F would be sigmoid or sigmoid_deriv
feed_forward(F, Weights, Inputs) ->
    F(dot_prod(Weights, Inputs)).

message_all(Procs, Message) ->
    lists:foreach(fun(P) -> P ! Message end, 
		  Procs).

perceptron(Weights, Inputs, Output_Pids) ->
    receive
	{learn, Backprop } ->
	    Learning_rate = 0.5,
	    
	    % calculate the correct sensitivities
	    New_sens = add_sensitivity(Sensitivities, Backprop),
	    Output_value = feed_forward(sigmoid, Weights, convert_to_values(Inputs)),
	    Derv_value = feed_forward(sigmoid_deriv, Weights, convert_to_values(Inputs)),
	    Sens = calculate_sensitivity(Backprop, Inputs, New_sens, 
					 Output_value, Derv_value),
	    io:format(" (~w) New Sensitivities: ~w ~n", [self(), New_sens]),
	    io:format(" (~w) Calculated Sensitivities: ~w ~n", [self(), Sens]),
	    
	    % Adjust all the weights 
	    Weight_adjustments = lists:map(fun(Input) ->
						   Learning_rate * Sens * Input
					   end,
					   conver_to_values(Inputs)),
	    New_weights = lists:zipwith(fun(X, Y) -> X + Y end, Weights, Weight_adjustments),
	    io:format(" (~w) Adjusted Weights: ~w ~n", [self(), Weights]),
	    

	    % propagate sensitivities and associated weights back 
	    % to the previous layer
	    
	    lists:zipwith(fun(Weight, Input_Pid) ->
				  Input_Pid ! {learn, {self(), Sens * Weight}}
			  end,
			  New_weights, 
			  convert_to_keys(Inputs)),

	    perceptron(New_weights, Inputs, New_sensitivities);
	
	{stimulate, Input} ->
	    % add Input to Intputs to get New Inputs
	    New_inputs = replace_input(Inputs, Input),
	    % calculate output of perceptron
	    Output = feed_forward(Weights, convert_to_list(New_inputs)),
	    
	    case Output_Pids of 
		[] ->  io:format("~w outputs: ~w~n", [self(), Output]);
		[_|_] -> message_all(Output_Pids, {stimulate, {self(), Output}})
	    end,
				  
	    % stimulate all the perceptrons I'm connected to with my response
	    perceptron(Weights, New_inputs, Output_Pids);  % this part handles state change

    % perceptron can double as source node
    % source node simply passes its input to its outputs
	{pass, Input_value} -> 
	    lists:foreach(fun(Output_PID) ->
				  io:format("Stimulating ~w with ~w ~n",
					    [Output_PID, Input_value]),
				  Output_PID ! {stimulate, {self(), Input_value}}
			  end,
			  Output_Pids);
	
	{connect_to_output, Receiver_Pid} ->
	    Combined_output = [Receiver_Pid | Output_Pids],
	    io:format("~w output connected to ~w: ~w~n", [self(), Receiver_Pid, Combined_output]),
	    perceptron(Weights, Inputs, Combined_output);

	{connect_to_input, Sender_Pid} ->
	    Combined_input = [ {Sender_Pid, 0.5} | Inputs],
	    io:format("~w inputs connected to ~w: ~w~n", [self(), Sender_Pid, Combined_input]),
	    perceptron([0.5 | Weights], Combined_input, Output_Pids)
    
    end.

% helper functions

% adds the propagating sensitivity to the Sensitivities Hash
add_sensitivity(Sensitivities, Backprop) when Sensitivities =/= [] ->
    replace_input(Sensitivities, Backprop);
add_sensitivity(Sensitivities, Backprop) when Sensitivities =:= [] ->    
    [].

% Calculates the sensitivity of this particular node
calculate_sensitivity(Backprop, Inputs, Sensitivities, Output_value, Derv_value) 
  when Sensitivities =/= [], Inputs =:= [] -> % When the node is an input node:
    null;
calculate_sensitivity(Backprop, Inputs, Sensitivities, Output_value, Derv_value) 
  when Sensitivities =:= [], Inputs =/= [] -> % When the node is an output node:
    {_, Training_value} = Backprop,
    (Training_value - Output_value) * Derv_value;
calculate_sensitivity(Backprop, Inputs, Sensitivities, Output_value, Derv_value) 
  when Sensitivities =/= [], Inputs =/= [] -> % When the node is a hidden node:
    Derv_value * lists:foldl(fun(E, T) -> E + T end, 0, convert_to_values(Sensitivities)).

% connects two perceptrons A and B together
connect(A, B) ->
    A ! {connect_to_output, B},
    B ! {connect_to_input, A}.
    
replace_input(Inputs, Input) ->
    {Input_Pid, _} = Input,
    lists:keyreplace(Input_Pid, 1, Inputs, Input).

convert_to_list(Inputs) ->
    {_Keys , Vals } = lists:unzip(Inputs),
    Vals.


%% example network

ex_net() ->
    N1 = spawn(neuralnet, perceptron, [[], [], []]),
    N2 = spawn(neuralnet, perceptron, [[], [], []]),		 
    N3 = spawn(neuralnet, perceptron, [[], [], []]),
    connect(N1, N2),
    connect(N1, N3).


    


    
    




    
	    
		     
