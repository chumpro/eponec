# Eponec

Eponec is a programming tool that allows the use of grammar to guide generative language models. Eponec uses a novel grammar specification system that is suitable for text generation.

The following are among the features of eponec:

- Labeled portions of text can be referenced within the same grammar specification and can be captured or used to change the structure of portions of the grammar.
- Eponec is ambiguous, a single piece of text can have multiple interpretations simultaneously. Choosing interpretations is controllable.
- Eponec grammars can mix operating on text and tokens allowing more control of the underlying language model.
- Eponec parsing can be nested, for example to reinterpret user input or generated text before making decisions based on it.
- Eponec can be extended to seamlessly work with networking, databases and other software.

Please have a look at the examples.

It is also worth mentioning that eponec does not reparse the text for each generated token. Internally the parser parses as far as it can and continues to do so whenever it is called during generation.

Eponec allows for parallelization of parsing, but this is not implemented yet.

Currently eponec contains the basic set of grammar components including Or, Group, Repeat, Regex, Text, Label, Use and Call in addition to some grammar components useful for dealing with tokens which are Token and TokensRepeat. It is also easy to create new components, for example for math, web and databases.

Eponec uses a context dictionary during parsing. This is where labeled data is set. The generate method of Parser objects returns this dictionary after generation, making the labeled data available to the caller.

Eponec is very simple, powerful and customizable.


### Working with text

- Or
  
For matching either or any of a list of components

		Or( 'A', 'B', 'C' )

- Group
  
Create an ordering.

		Group( 'This ', 'is ', 'an ', 'example.' )

- Repeat
  
Allow for repetition.

		Repeat( 'Ha' )

- Regex

For matching using regular expressions.

 		Regex( r'[a-z ]+' )

- Text

For matching a whole string.

 		Text( 'Hello world!' )

- Label

To name a portion of generated text.

 		Label( 'name', Regex( r'[A-Z][a-z]*' ) )

- Use

To refer to a labeled portion of text.

 		Use( 'name' )

- Call

To call a function that has access to the context, ie. labeled portions of text, and generate output. The called function can change the grammar based on the context by returning a Parser object.

		def f( context ) :

			context[ 'x' ] = int( context[ 'a' ] ) + int( context[ 'b' ] )

			return str( context[ 'x' ] )	# A: One can return new Parser objects, B: 'x' will exist in the context going forward

		Call( f )


### Working with tokens directly

- Token

Matches a single token by id.

  	Token( tokenizer.eos_token_id )

- TokensRepeat

Matches from a list of tokens repeating over a continuous area in the text given a boolean tensor or list of token ids.

 	TokensRepeat( 1001, 1002, 1003 )


### Odd and useful stuff

- Sample

Sample k components and match them as an Or.

	Sample( 'Luckily', 'Sadly', 'Surprisingly', 'As expected', k = 2 )

- Input

Read user input. The user input becomes part of the text. Please see the examples.

 	Input( prompt = '> ', True )

- Eos

Matches tokenizer.eos_token_id. Used to end parsing.

	Eos()


The Parser objects are nested to create more complex parsers. To make parsers easier on the eyes, a string can be used in place of Parser objects, so that 'asdf' becomes Text( 'asdf' ) automatically.

	Repeat( 'asdf' )

is equivalent to

	Repeat( Text( 'asdf' ) )

In the same manner, multiple components are interpreted as a Group.

	Repeat( 'asdf', 'qwer' )

is equivalent to

	Repeat( Group( 'asdf', 'qwer' ) )


In addition to the grammar components, eponec has some utility functions.


## Installation

Eponec is tested in a Python 3.10 conda environment with pytorch and transformers. Please read the instructions of the individual packages. The following sets up the conda environment I use for development ( no GPU ).

	conda create myenv python=3.10

	conda activate myenv

	pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

	pip install git+https://github.com/huggingface/transformers

	pip install accelerate
	pip install git+https://github.com/huggingface/diffusers

	git clone https://github.com/chumpro/eponec.git
	cd eponec
	pip install .



## Programming style

The eponec parsing mechanism is defined by the classes Parser and Matcher. To understand these please read the source code.


## One little quirk

Due to the way matching ( regular expressions ) work, it is not possible to determine if you have matched all the data in the general case. For example with '[a-z]+' the string 'abcd' cannot be known to be the whole match. An 'e' could come along, then the match would be 'abcde'. Once there is something other than 'a' to 'z' it can be determined what the whole match is, eg. 'abcde '. This is important for capturing with Label. A decision has been made to use longest possible match.

For example:

	Group(
		Label( 'data', Regex( r'[a-z]+' ) ),
		Text( '.' ),
		Use( 'data' )
	)

Here '.' is a "delimiter" and is necessary for 'data' to be available when it is used. It is also necessary to use a delimiter before Eos() at the end of a full parser. Typically Text is used as the delimiter.

There is a special case, escaped strings, Text. Techinically those patterns can be matched with shortest match. However since other components will depend on this behaviors, and you still want to use regular expressions with longest match, Text is set to behave like Regex in this respect. In the following example the '.' delimiter is still necessary.

	Group(
		Label( 'data', Text( 'asdf' ) ),
		Text( '.' ),
		Use( 'data' )
	)

So basically, put something between Label and whatever relies on that labeled data.



## Performance

Performance. I am writing this on a laptop without a GPU. Generation slows down quite a bit with larger prompts. It may be possible to correct this, but it is recommended to use eponec on a computer with a capable GPU.


## Bugs

Nothing I know about, but all code paths have not been tested thoroughly. Especially the ability to use eponec outside text generation, that is a as a ordinary parser of constant strings.


## TODO

- Parallelization
- Sample


## Contributions

Feel free to contribute and give suggestion. I will focus on bug fixing, but all suggestions are very welcome.

Monero: 48hRRk1mN9nVLVrUL8e7qLNznEXzvsXzqWW24jurw4QAcaTwUa1K9FwGkzwkfGFNSzL58zNPLEGBeet6ELJ5VZ4zKDjdqiP

