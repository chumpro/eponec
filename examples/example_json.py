
import eponec
from eponec import Group, Repeat, Text, Regex, Input, Call, Eos, Or, Nothing


# NOTE: This has been a struggle to test with the TinyLlama model. It may not be working properly.

# NOTE: Notice the use of Call as a reference to allow cycles. I think it's good:)


eponec.init_llama()


json_root = Call( lambda context : json_complex )


json_integer = Regex( r'[+-]?[0-9]+' )

json_float = Regex( r'[+-]?[0-9]*[.][0-9]+' )

json_string = Regex( r'"[a-zA-Z0-9 (),.;:[\]]*"' )

json_list_components = Group(
	json_root,
	Or(
		Group(
			',',
			Call( lambda context : json_list_components )
		),
		Nothing()
	)
)

json_list = Group( '[', json_list_components, ']' )

json_pair = Group( json_string, ':', json_root )

json_dict_components = Group(
	json_pair,
	Or(
		Group(
			',',
			Call( lambda context : json_dict_components )
		),
		Nothing()
	)
)

json_dict = Group( '{', json_dict_components, '}' )

json_complex = Or(

	json_integer,
	json_float,
	json_string,
	json_list,
	json_dict,

)


parser = Repeat(
	Group(
		Input( '> ' ),
		': ',
		json_root,
		'\n\n'
	)
)


parser.generate( '\n', verbose = True )


#

