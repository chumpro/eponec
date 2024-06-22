

import eponec
from eponec import Group, Repeat, Text, Regex, Input, TokensRepeat


eponec.init()


parser = Repeat(
	Group(
		Input( '> ', False ),
		TokensRepeat( eponec.regex_token_match( r'[a-zA-Z ,":;]+' ) ),
		'. ',
	)
)


parser.generate( 'The following is a short story: ', verbose = True )


#

