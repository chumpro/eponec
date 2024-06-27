

import regex
import random

import concurrent.futures

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LogitsProcessor, LogitsProcessorList



config = None


def init_llama( model_id = None, local_files_only = True ) :

	global config

	model_id = 'TinyLlama/TinyLlama_v1.1' if model_id is None else model_id

	config = {}

	config[ 'model_id' ] = model_id

	tokenizer = AutoTokenizer.from_pretrained(
		config[ 'model_id' ],
		legacy = False,
		local_files_only = local_files_only,
		clean_up_tokenization_spaces = False,
		add_bos_token = False,
		add_eos_token = False,
	)

	model = AutoModelForCausalLM.from_pretrained( config[ 'model_id' ], local_files_only = local_files_only )

	config[ 'tokenizer' ] = tokenizer

	config[ 'model' ] = model

	config[ 'vocab_size' ] = len( config[ 'tokenizer' ] )

	def get_usable_vocab() :

		vocab = { v : fix_token( k ) for k, v in config[ 'tokenizer' ].get_vocab().items() }

		return vocab

	def fix_token( s ) :

		return s.replace( chr( 9601 ), ' ' ).replace( '<0x0A>', '\n' )

	config[ 'usable_vocab' ] = get_usable_vocab()

	config[ 'empty_logits_mask' ] = torch.zeros( config[ 'vocab_size' ], dtype = torch.bool )

	def fix_output( s ) :

		return s.replace( '</s> ', '</s>' )

	config[ 'fix_output' ] = fix_output

	def encode( string ) :

		output = config[ 'tokenizer' ]( string, return_tensors = "pt", return_offsets_mapping = True, add_special_tokens = False )

		return output[ 'input_ids' ][ 0 ], output[ 'offset_mapping' ][ 0 ]

	config[ 'encode' ] = encode

	def decode( token_ids ) :

		string = config[ 'tokenizer' ].decode( token_ids )

		return string

	config[ 'decode' ] = decode






def logits_mask_new( *token_ids ) :

	if len( token_ids ) == 0 :

		return config[ 'empty_logits_mask' ]

	token_ids = torch.tensor( token_ids )

	logits_mask = config[ 'empty_logits_mask' ].clone()

	if len( token_ids ) > 0 :

		logits_mask[ token_ids ] = True

	return logits_mask


def regex_token_match( pattern, prefix = '' ) :

	if isinstance( pattern, str ) :

		pattern = regex.compile( pattern )

	token_ids = []

	for token_id, token_string in config[ 'usable_vocab' ].items() :

		s = prefix + token_string

		m = pattern.fullmatch( s, partial = True )

		if m is not None :

			token_ids.append( token_id )

	return tuple( token_ids )


def regex_token_match( pattern, prefix = '' ) :					# NOTE: This is parallell version of previous function. Not properly tested.

	if isinstance( pattern, str ) :

		pattern = regex.compile( pattern )

	def chunks( l, n ):

		for i in range( 0, len( l ), n ) :

		    yield l[ i : i + n ]

	def do_matches( c ) :

		token_ids = []

		for token_id, token_string in c :

			s = prefix + token_string

			m = pattern.fullmatch( s, partial = True )

			if m is not None :

				token_ids.append( token_id )

		return token_ids

	token_ids = []

	with concurrent.futures.ThreadPoolExecutor() as executor :

		futures = [ executor.submit( do_matches, c ) for c in chunks( list( config[ 'usable_vocab' ].items() ), 20000 ) ]

		for future in concurrent.futures.as_completed( futures ) :

			token_ids += future.result()

	return tuple( token_ids )


def find_token_offset( offset, offset_mapping ) :

	for token_pos, om in enumerate( offset_mapping ) :

		if om[ 0 ].item() == offset :

			return token_pos


def match_forward_logits_mask( start_pos, logits_mask, input_ids ) :

	for token_pos, token_id in enumerate( input_ids[ start_pos : ] ) :

		if logits_mask[ token_id ] != True :

			return start_pos + token_pos




# NOTE: The function 'parse_many' is the workhorse that binds the stuff together. If you write your own Parser subclasses, you will likely want to use it.


def parse_many( context, in_matchers ) :										# NOTE: This can also be parallellized.

	matchers = set()

	offsets = set()

	logits_mask = logits_mask_new()

	for matcher in in_matchers :

		new_matchers, new_offsets, new_logits_mask = matcher.parse( context )

		matchers |= new_matchers

		offsets |= new_offsets

		logits_mask = logits_mask | new_logits_mask

	return matchers, offsets, logits_mask



class LogitsMaskProcessor ( LogitsProcessor ) :

	def __init__( self, logits_mask ) :

		self.logits_mask = logits_mask

	def __call__( self, input_ids, scores ) :

		scores += torch.where( self.logits_mask, 0, -float( 'inf' ) )

		return scores




# NOTE: Here comes the parser. A Parser "attaches" a Matcher to the string. The Matcher sits there until it is finnished.
# NOTE: Parser is subclassed to make new types of parsering possible.


class Matcher ( object ) :

	def __init__( self, parser, offset, *state ) :

		self.parser = parser
		self.offset = offset
		self.state = state

	def parse( self, context ) :

		return self.parser.parse( self.offset, context, *self.state )

	def __hash__( self ) :

		return hash( ( type( self ), self.parser, self.offset ) )

	def __eq__( self, other ) :

		if isinstance( other, type( self ) ) :

			return ( self.parser == other.parser ) and ( self.offset == other.offset ) and ( self.state == other.state )



class Parser ( object ) :

	def __init__( self, *structure, **kwargs ) :

		self.structure = structure

		self.kwargs = kwargs

		self.store = None

	def attach( self, offset, *state ) :					# NOTE: The method 'attach' share the last part of signature you decide for 'parse'.

		return Matcher( self, offset, *state )

	def parse( self, offset, context, *state ) :			# NOTE: You decide the last part ( state ) of the method signature.

		raise NotImplemented()

		return parsers, offsets, logits_mask

	def __hash__( self ) :

		return hash( ( type( self ), self.structure, self.store ) )

	def __eq__( self, other ) :

		if isinstance( other, type( self ) ) :

			return ( self.structure == other.structure ) and ( self.store == other.store )


	def generate( self, prompt = None, context = None, verbose = False, keep_going = False ) :

		context = {} if context is None else context

		context[ 'prompt' ] = ' ' if prompt is None else prompt

		matchers = [ self.attach( len( context[ 'prompt' ] ) ) ]

		while True :

			context[ 'terminals' ] = []

			if verbose :

				print( [ context[ 'prompt' ] ] )

			prompt = context[ 'prompt' ]

			input_ids, offset_mapping = config[ 'encode' ]( prompt )

			context[ 'input_ids' ] = input_ids
			context[ 'offset_mapping' ] = offset_mapping

			matchers, new_offsets, logits_mask = parse_many( context, matchers )

			if ( len( context[ 'terminals' ] ) == 1 ) and isinstance( context[ 'terminals' ][ 0 ], str )  :

				context[ 'prompt' ] += context[ 'terminals' ][ 0 ]

				continue

			allowed_token_count = logits_mask.count_nonzero()

			if allowed_token_count == 0 :

				if verbose :

					print( 'FAILED: No allowed tokens.' )

				return None

			logits_processor = LogitsMaskProcessor( logits_mask )

			output_ids = config[ 'model' ].generate(
				input_ids.unsqueeze( 0 ),
				attention_mask = torch.ones_like( input_ids ).unsqueeze( 0 ),
				logits_processor = LogitsProcessorList( [ logits_processor ] ),
				do_sample = True,
				max_new_tokens = 1,
				temperature = 0.9,
			)[ 0 ]


			if output_ids[ -1 ] == config[ 'tokenizer' ].eos_token_id and ( not keep_going ) :

				if verbose :

					print( 'SUCCESS: Eos encountered.' )

				return context

			context[ 'prompt' ] = config[ 'fix_output' ]( config[ 'decode' ]( output_ids ) )



# NOTE: The rest of this code is all implementations of different parsers, subclasses of Parse that only implement the 'parse' method.
# NOTE: They may look a little nasty but this design leaves all the good stuff in the 'parse' method of Parser subclasses.


class Regex ( Parser ) :

	def parse( self, offset, context ) :

		[ pattern ] = self.structure

		if isinstance( pattern, str ) :

			pattern = regex.compile( pattern )

		m = pattern.fullmatch( context[ 'prompt' ][ offset : ], partial = True )


		if m is None :

			return set(), set(), logits_mask_new()


		token_ids = regex_token_match(
			pattern,
			prefix = context[ 'prompt' ][ offset : ]
		)

		logits_mask = logits_mask_new( *token_ids )

		context[ 'terminals' ].append( None )

		if m.partial :

			return set( [ self.attach( offset ) ] ), set(), logits_mask

		else :

			return set( [ self.attach( offset ) ] ), set( [ len( context[ 'prompt' ] ) ] ), logits_mask



class Text ( Parser ) :

	def parse( self, offset, context ) :

		def chrs( s ) :

			return [ x for x in s ]

		[ text ] = self.structure

		pattern = regex.compile( regex.escape( text ) )

		m = pattern.fullmatch( context[ 'prompt' ][ offset : ], partial = True )

		if m is None :

			return set(), set(), logits_mask_new()


		token_ids = regex_token_match(
			pattern,
			prefix = context[ 'prompt' ][ offset : ]
		)

		logits_mask = logits_mask_new( *token_ids )

		terminal = text[ len( context[ 'prompt' ] ) - offset : ]
		
		if len( terminal ) > 0 :

			context[ 'terminals' ].append( terminal )

		if m.partial :

			return set( [ self.attach( offset ) ] ), set(), logits_mask

		else :

			return set( [ self.attach( offset ) ] ), set( [ len( context[ 'prompt' ] ) ] ), logits_mask





class Nothing ( Parser ) :

	def parse( self, offset, context ) :

		return set(), set( [ offset ] ), logits_mask_new()


class Group ( Parser ) :

	def parse( self, offset, context, matchers_1 = None, matchers_2 = None ) :

		[ p0, *ps ] = self.structure

		if isinstance( p0, str ) :

			p0 = Text( p0 )

		p1 = Nothing() if ( len( ps ) == 0 ) else Group( *ps )

		matchers_1 = set( [ p0.attach( offset ) ] ) if matchers_1 is None else matchers_1

		matchers_2 = set() if matchers_2 is None else matchers_2

		matchers_new_1, offsets_new_1, logits_mask_new_1 = parse_many( context, matchers_1 )

		matchers_2 |= set( [ p1.attach( o ) for o in offsets_new_1 ] )

		matchers_new_2, offsets_new_2, logits_mask_new_2 = parse_many( context, matchers_2 )

		logits_mask = logits_mask_new_1 | logits_mask_new_2

		if ( len( matchers_new_1 ) == 0 ) and ( len( matchers_new_2 ) == 0 ) :

			matchers_result = set()

		else :

			matchers_result = set( [ self.attach( offset, matchers_new_1, matchers_new_2 ) ] )

		return matchers_result, offsets_new_2, logits_mask


class Or ( Parser ) :

	def parse( self, offset, context, matchers = None ) :

		matchers = set( [ p.attach( offset ) for p in [ Text( p ) if isinstance( p, str ) else p for p in self.structure ] ] ) if matchers is None else matchers

		matchers_new, offsets_new, logits_mask = parse_many( context, matchers )

		if len( matchers_new ) == 0 :

			matchers_result = set()

		else :

			matchers_result = set( [ self.attach( offset, matchers_new ) ] )

		return matchers_result, offsets_new, logits_mask


class Repeat ( Parser ) :

	def parse( self, offset, context, matchers = None ) :

		if len( self.structure ) > 1 :

			parser = Group( *self.structure )

		else :

			[ parser ] = self.structure
	
			if isinstance( parser, str ) :

				parser = Text( parser )

		matchers = set( [ parser.attach( offset ) ] ) if matchers is None else matchers

		offsets = set()

		new_matchers = matchers

		while True :

			matchers, new_offsets, logits_mask = parse_many( context, matchers )

			ol = len( offsets )

			offsets |= new_offsets

			if len( offsets ) == ol :

				break

			matchers |= set( [ parser.attach( o ) for o in new_offsets ] )

		if len( matchers ) == 0 :

			matchers_result = set()

		else :

			matchers_result = set( [ self.attach( offset, matchers ) ] )

		return matchers_result, offsets, logits_mask


class Label ( Parser ) :

	def parse( self, offset, context, matchers = None, value = None ) :

		# todo: only initialize if matchers == None

		[ label, *parsers ] = self.structure

		if len( parsers ) == 0 :

			parser = Nothing()

		elif len( parsers ) == 1 :

			[ parser ] = parsers

		else :

			parser = Group( *parsers )

		if isinstance( parser, str ) :

			parser = Text( parser )


		matchers = set( [ parser.attach( offset ) ] ) if matchers is None else matchers

		matchers, offsets, logits_mask = parse_many( context, matchers )


		if len( matchers ) == 0 :

			context[ label ] = value

			matchers_result = set()

		else :

			value = ' '.join( context[ 'prompt' ][ offset : ].split() )

			matchers_result = set( [ self.attach( offset, matchers, value ) ] )

		return matchers_result, offsets, logits_mask


class Use ( Parser ) :

	def parse( self, offset, context ) :

		[ label ] = self.structure

		if label in context :

			return Text( context[ label ] ).attach( offset ).parse( context )

		else :

			return set( [ self.attach( offset ) ] ), set(), logits_mask_new()


class Call ( Parser ) :

	def parse( self, offset, context, matchers = None ) :

		[ function ] = self.structure

		result = function( context )

		if result is None :

			result = ''

		if isinstance( result, str ) :

			result = Text( result )

		return result.attach( offset ).parse( context )


class Input ( Parser ) :

	def parse( self, offset, context ) :

		[ input_prompt ] = self.structure

		return Text( input( input_prompt ) ).attach( offset ).parse( context )


class Token ( Parser ) :

	def parse( self, offset, context ) :

		[ token_id ] = self.structure

		if len( context[ 'prompt' ] ) > offset :

			token_pos = find_token_offset( offset, context[ 'offset_mapping' ] )

			print( 'token_pos', token_pos )

			if token_pos is None :

				return set(), set(), logits_mask_new()

			else :

				if context[ 'input_ids' ][ token_pos ] == token_id :

					return set(), set( [ context[ 'offset_mapping' ][ token_pos ][ 1 ].item() ] ), logits_mask_new()

				else :

					return set(), set(), logits_mask_new()

		elif len( context[ 'prompt' ] ) == offset :

			context[ 'terminals' ].append( None )

			return set( [ self.attach( offset ) ] ), set(), logits_mask_new( token_id )

		else :

			return set( [ self.attach( offset ) ] ), set(), logits_mask_new()


class Eos( Parser ) :

	def parse( self, offset, context ) :

		return Token( config[ 'tokenizer' ].eos_token_id ).attach( offset ).parse( context )


class TokensRepeat ( Parser ) :

	def parse( self, offset, context ) :

		[ logits_mask ] = self.structure

		if isinstance( logits_mask, tuple ) :

			logits_mask = logits_mask_new( *logits_mask )

		if len( context[ 'prompt' ] ) == offset :

			context[ 'terminals' ].append( None )

			return set( [ self.attach( offset ) ] ), set(), logits_mask

		elif len( context[ 'prompt' ] ) > offset :

			start_pos = find_token_offset( offset, context[ 'offset_mapping' ] )

			if start_pos is None :

				return set(), set(), logits_mask_new()

			else :

				stop_pos = match_forward_logits_mask( start_pos, logits_mask, context[ 'input_ids' ] )

				if stop_pos is None :

					context[ 'terminals' ].append( None )

					return set( [ self.attach( offset ) ] ), set( [ len( context[ 'prompt' ] ) ] ), logits_mask

				else :

					stop_offset = context[ 'offset_mapping' ][ stop_pos ][ 1 ].item()

					return set(), set( [ stop_offset ] ), logits_mask_new()

		else :

			return set( [ self.attach( offset ) ] ), set(), logits_mask_new()
		

class Sample ( Parser ) :

	def parse( self, offset, context ) :

		if 'k' in self.kwargs :

			k = self.kwargs[ 'k' ]

		else :

			k = 1

		return Or( *random.sample( self.structure, k = k ) ).attach( offset ).parse( context )








#

