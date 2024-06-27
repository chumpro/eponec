

import eponec
from eponec import Group, Repeat, Regex, Input, TokensRepeat


eponec.init_llama( 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' )


class Chat ( eponec.Parser ) :

	def parse( self, offset, context ) :

		[ role, user_parser, assistant_parser ] = self.structure

		return Group(
			'<|system|>\n', role, '</s>\n',
			Repeat(
				'<|user|>\n', user_parser, '</s>\n',
				'<|assistant|>\n', assistant_parser, '</s>\n',
			)
		).attach( offset ).parse( context )


parser = Chat(
	Input( 'role> ' ),
	Input( 'user> ' ),
	TokensRepeat( eponec.regex_token_match( r'[a-zA-Z0-9 .,;:"()[\]\']+' ) ),
)


parser.generate( ' ', verbose = True, keep_going = True )


#

