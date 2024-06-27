

import eponec
from eponec import Group, Repeat, Text, Regex, Input, TokensRepeat, Or, Label, Use, Sample, Eos


eponec.init_llama()


sentence_logits_mask = eponec.regex_token_match( r'[a-zA-Z ,":;]+' )


parser = Group(
	'Once upon a time there was a ',
	Label( 'role',
		Sample( 'rich', 'poor', 'weak', 'mighty', 'cowardly', 'brave', 'good', 'evil', k = 2 ),
		' ',
		Sample( 'knight', 'maid', 'thief', 'chef', 'smith', 'noble', 'mage', 'wizard', 'witch', k = 2 ),
	),
	' called ',
	Label( 'name', Regex( r'[A-Z][a-z]{4,7}' ) ),
	'. ',
	Label( 'early_life',
		Sample( 'One night ', 'As a child ', 'A few years ago ', 'One time ' ),
		Use( 'name' ),
		eponec.TokensRepeat( sentence_logits_mask ),
		'. '
	),
	Label( 'recent_event',
		Sample( 'Recently', 'Yesterday', 'Currently' ),
		eponec.TokensRepeat( sentence_logits_mask ),
		'. '
	),
	Label( 'trivia',
		Sample( 'Luckily', 'Surprisingly', 'Sadly', 'Obviously' ),
		eponec.TokensRepeat( sentence_logits_mask ),
		'. '
	),
	'The end.',	# Note: Neccesary delimiter for 'trivia' to appear in the result.
	Eos(),
)

result = parser.generate( 'The following is a short story: ', verbose = True )

print( 'result', result )


#

