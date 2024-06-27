

import eponec
from eponec import Group, Repeat, Text, Regex, Input, TokensRepeat, Label, Call, Eos

import random


eponec.init_llama()


class RandomInteger ( eponec.Parser ) :

	def parse( self, offset, context ) :

		[ left, right ] = self.structure

		x = random.randint( left, right )

		return Text( str( x ) ).attach( offset ).parse( context )


parser = Group(
	Label( 'a', RandomInteger( 0, 1000 ) ),
	' + ',
	Label( 'b', RandomInteger( 0, 100 ) ),
	' = ',
	Label( 'x', Regex( r'[0-9]{1,8}' ) ),
	',',
	Eos()
)


total = 10

count = 0

for i in range( 1, total ) :

	result = parser.generate( 'Here are some simple equations: ' )

	a = int( result[ 'a' ] )
	b = int( result[ 'b' ] )
	x = int( result[ 'x' ] )

	if a + b == x :

		count += 1

		print( a, ' + ', b, ' == ', x, ' OK' )

	else :

		print( a, ' + ', b, ' == ', x, ' FAIL' )


print( 'Success rate: ', count / total )


#

