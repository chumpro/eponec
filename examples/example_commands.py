

import eponec
from eponec import Group, Repeat, Text, Regex, Input, Label, Or, Call


eponec.init()


def command_excamine( context ) :

	print( '\n*** Excamining', context[ 'subject' ] )

	return ''

def command_go_to( context ) :

	print( '\n*** Moving to', context[ 'location' ] )

	return ''

def command_pick_up( context ) :

	print( '\n*** Picking up', context[ 'item' ] )

	return ''


prompt = ' '.join( """

	A simpler and more concise way to say "get the hammer" is "pick up the hammer".

	A simpler and more concise way to say "leave for the beach" is "go to the beach".

	A simpler and more concise way to say "zoom in on the umbrella" is "excamine umbrella".

""".split() )


parser = Repeat(
	Group(
		'A simpler and more concise way to say "',
		Label( 'input', Input( '> ', False ) ),
		'" is "',
		Or(
			Group( 'excamine', Label( 'subject', Regex( r'[a-z ]+' ) ), '"', Call( command_excamine ) ),
			Group( 'go to', Label( 'location', Regex( r'[a-z ]+' ) ), '"', Call( command_go_to ) ),
			Group( 'pick up', Label( 'item', Regex( r'[a-z ]+' ) ), '"', Call( command_pick_up ) ),
		),
		'. ',
	)
)

parser.generate( prompt, verbose = True )


#

