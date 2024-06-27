

import eponec
from eponec import Group, Repeat, Text, Regex, Input, TokensRepeat


eponec.init_llama()


parser = Repeat(
	Group(
		Input( '> ' ),
		TokensRepeat( eponec.regex_token_match( r'[a-zA-Z ,":;]+' ) ),
		'. ',
	)
)


parser.generate( 'The following is a short story: ', verbose = True )


#

